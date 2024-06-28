import os
from argparse import Namespace
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from accelerate import Accelerator
from tabulate import tabulate
from torch.utils.data import DataLoader

import wandb
from src.networks.multimodal import FlexibleFusion
from src.networks.unimodal import FFN, GNN, build_vgg19_encoder
from src.tools.evaluation import evaluate


# --- General ---
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    # Required for determenistic behaviour on graph conv layers
    # Note that this has a significant performance impact on CUDA
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def mkdir(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_model(model: nn.Module, opt: Namespace, predictions: pd.DataFrame) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt,
        },
        os.path.join(mkdir(f"{opt.ckpt_dir}"), f"{opt.model}_{opt.k}.pt"),
    )
    predictions.to_csv(
        os.path.join(mkdir(f"{opt.ckpt_dir}/results"), f"{opt.model}_{opt.k}.csv")
    )


# --- Loss ---
class PathomicLoss(nn.Module):
    def __init__(self, opt: Namespace) -> None:
        """Loss function for survival and grade prediction tasks."""

        super().__init__()
        self.nll = False if opt.task == "surv" else True
        self.cox = False if opt.task == "grad" else True
        self.l1 = opt.l1
        self.device = opt.device

    @staticmethod
    def cox_loss(
        survtime: torch.Tensor, event: torch.Tensor, hazard_pred: torch.Tensor
    ) -> torch.Tensor:
        """Source: https://github.com/traversc/cox-nnet"""
        # Predictions are not independent in Coxloss; calculating over batch != whole dataset
        R_mat = (survtime.repeat(len(survtime), 1) >= survtime.unsqueeze(1)).int()
        theta = hazard_pred.view(-1)
        exp_theta = theta.exp()
        loss_cox = -torch.mean((theta - (exp_theta * R_mat).sum(dim=1).log()) * event)
        return loss_cox

    def forward(
        self,
        model: nn.Module,
        grade: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
        grade_pred: torch.Tensor,
        hazard_pred: torch.Tensor,
    ) -> torch.Tensor:
        _zero = torch.tensor(0.0).to(self.device)
        nll = F.nll_loss(grade_pred, grade) if self.nll else _zero
        cox = self.cox_loss(time, event, hazard_pred) if self.cox else _zero
        l1 = model.l1() * self.l1 if self.l1 else _zero
        loss = nll + cox
        return loss + l1, loss


# --- Training ---
def define_model(opt: Namespace) -> nn.Module:
    """Defines the model architecture based on command line arguments."""

    # Define graph pooling strategy
    if "graph" in opt.model:
        opt.graph_pool = None
        if opt.mil == "local" or 'qbt' in opt.model:
            opt.graph_pool = "collate"
        elif opt.mil == "global":
            opt.graph_pool = opt.pool

    # Define local pooling strategy
    local_pool = opt.pool if opt.mil == "local" else None

    if opt.model == "omic":
        model = FFN(xdim=320 if opt.rna else 80, fdim=32, dropout=opt.dropout)

    elif opt.model == "graph":
        model = GNN(fdim=32, pool=opt.graph_pool, dropout=opt.dropout, local=local_pool)

    elif opt.model == "path":
        if opt.mil != "PFS":
            raise NotImplementedError("MIL not implemented for VGG model.")
        model = build_vgg19_encoder(fdim=32)

    elif any(m in opt.model for m in ("pathomic", "graphomic", "pathgraphomic")):
        model = FlexibleFusion(opt, fdim=32, mmfdim=32 if "qbt" in opt.model else 64)
    else:
        raise NotImplementedError(f"Model {opt.model} not implemented")
    return model


def define_scheduler(
    opt: Namespace, optimizer: torch.optim.Optimizer
) -> lr_scheduler._LRScheduler:
    """Defines the linearly decaying learning rate scheduler used in the original paper."""

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.lr_fix) / float(opt.n_epochs - opt.lr_fix)
        return lr_l

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # return torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.008, total_steps=850)


def make_cv_results_table(metrics_list: list) -> str:
    """Makes a table summarising CV performance."""

    df = pd.DataFrame(
        metrics_list,
        columns=["Accuracy", "AUC", "C-Index"],
        index=range(1, len(metrics_list) + 1),
    )
    df.index.name = "Fold"
    df = df.loc[:, (df != 0).any(axis=0)]
    df.loc["mean"] = df.mean()
    df.loc["std"] = df.std()
    return tabulate(df.T, headers="keys", tablefmt="rounded_grid", floatfmt=".3f")


def init_accelerator(opt: Namespace) -> Accelerator:
    """Initialises an accelerator on the correct device."""

    # Only use MPS for path
    cpu = False if opt.model == "path" or torch.cuda.is_available() else True
    accelerator = Accelerator(cpu=cpu, step_scheduler_with_optimizer=False)
    opt.device = accelerator.device
    print(f"Device: {opt.device}")
    return accelerator


def configure_wandb(opt: Namespace, k: int) -> "wandb.Run":
    """Initialise a WandB run that tracks a single fold as part of a CV group."""

    os.environ["WANDB_SILENT"] = "true"
    return wandb.init(
        reinit=True,
        mode="disabled" if opt.dry_run else "online",
        project="M3IL",
        config=opt,
        group=opt.group,
        name=f"{opt.group}_{k}",
    )


def assign_ckptdir_and_group(opt: Namespace) -> None:
    """Assigns checkpoint directory and CV group for WandB. Updates opt in place."""

    rna = "_rna" if (opt.rna and "omic" in opt.model) else ""
    pool = f"_{opt.pool}" if opt.pool != "mean" else ""
    # Ignore MIL for omic as there is only 1 instance per patient
    if opt.model == "omic":
        opt.ckpt_dir = f"{opt.save_dir}/{opt.task}/{opt.model}{rna}"
        opt.group = f"{opt.task}_{opt.model}{rna}"
    else:
        opt.ckpt_dir = f"{opt.save_dir}/{opt.task}/{opt.model}{rna}_{opt.mil}{pool}"
        opt.group = f"{opt.task}_{opt.model}{rna}_{opt.mil}{pool}"
    print(f"Checkpoint dir: ./{opt.ckpt_dir}/")


def log_fold(cv_metrics: list, fold_metrics: Dict, k: int) -> list:
    """Print a table summarising fold performance and update running CV list."""

    fold_table = [[f"Split {k}", "Loss", "Accuracy", "AUC", "C-Index"]]
    for split, mdict in fold_metrics.items():
        fold_table.append(
            [split, mdict["loss"], mdict["accuracy"], mdict["auc"], mdict["c_indx"]]
        )

    tkwgs = {"headers": "firstrow", "tablefmt": "rounded_grid", "floatfmt": ".3f"}
    print(tabulate(fold_table, **tkwgs))

    mtest = fold_metrics["Test"]
    cv_metrics.append([mtest["accuracy"], mtest["auc"], mtest["c_indx"]])

    return cv_metrics


def log_epoch(
    opt: Namespace,
    model: nn.Module,
    epoch: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    train_loss: float,
    all_preds: Dict,
) -> str:
    """Log epoch performance to WandB and return a description string for tqdm."""

    # Use precomputed predictions/loss for train
    mtrain = evaluate(model, train_loader, loss_fn, opt, pd.DataFrame(all_preds))
    train_acc = mtrain["accuracy"]
    train_auc = mtrain["auc"]
    c_train = mtrain["c_indx"]

    mtest = evaluate(model, test_loader, loss_fn, opt)
    test_loss = mtest["loss"]
    test_acc = mtest["accuracy"]
    test_auc = mtest["auc"]
    c_test = mtest["c_indx"]

    desc = f"Epoch {epoch} (train/test) | Loss: {train_loss:.2f}/{test_loss:.2f} | "
    wandb.log({"train_loss": train_loss, "test_loss": test_loss})
    if opt.task != "surv":
        desc += f"Acc: {train_acc:.2f}/{test_acc:.2f} | "
        desc += f"AUC: {train_auc:.2f}/{test_auc:.2f} | "
        wandb.log(
            {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_auc": train_auc,
                "test_auc": test_auc,
            }
        )
    if opt.task != "grad":
        desc += f"C-Index: {c_train:.2f}/{c_test:.2f} | "
        wandb.log({"c_train": c_train, "c_test": c_test})

    return desc
