import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tabulate import tabulate

import wandb
from src.networks.multimodal import FlexibleFusion
from src.networks.unimodal import FFN, GNN, build_vgg19_encoder
from src.tools.evaluation import evaluate

# from line_profiler import profile


# Predictions are not independent in Coxloss; calculating over batch != whole dataset
class PathomicLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.nll = 0 if opt.task == "surv" else 1
        self.cox = 0 if opt.task == "grad" else 1
        self.l1 = opt.l1

    @staticmethod
    def l1_reg(model):
        # We only regularise the omic_net
        if hasattr(model, "omic_net"):
            return sum(torch.abs(W).sum() for W in model.omic_net.parameters())
        else:
            # If no omic_net, assume opt.l1 set to 0
            # This is here in case we are training the omic net itself
            return sum(torch.abs(W).sum() for W in model.parameters())

    @staticmethod
    def cox_loss(survtime, event, hazard_pred):
        R_mat = (survtime.repeat(len(survtime), 1) >= survtime.unsqueeze(1)).int()
        theta = hazard_pred.view(-1)
        exp_theta = theta.exp()
        loss_cox = -torch.mean((theta - (exp_theta * R_mat).sum(dim=1).log()) * event)
        return loss_cox

    def forward(self, model, grade, time, event, grade_pred, hazard_pred):
        _zero = torch.tensor(0.0).to(grade.device)
        nll_loss = F.nll_loss(grade_pred, grade) if self.nll else _zero
        cox_loss = self.cox_loss(time, event, hazard_pred) if self.cox else _zero
        l1_loss = self.l1_reg(model) if self.l1 else _zero
        total_loss = nll_loss + cox_loss + l1_loss * self.l1
        return total_loss


# --- Training ---
def define_model(opt):

    # TODO: Tidy this up
    if 'graph' in opt.model:
        agg = None
        if opt.mil == "pat":
            if 'qbt' in opt.model:
                agg = 'collate'
            else:
                agg = 'attn' if opt.attn_pool else 'mean'
        opt.graph_pool = agg

    if opt.model == "omic":
        model = FFN(xdim=320 if opt.rna else 80, fdim=32, dropout=opt.dropout)

    elif opt.model == "graph":

        model = GNN(fdim=32, pool=opt.graph_pool, dropout=opt.dropout)

    elif opt.model == "path":
        if opt.mil == 'pat':
            raise NotImplementedError("Bagging not implemented for path model.")
        model = build_vgg19_encoder(fdim=32)

    elif any(m in opt.model for m in ("pathomic", "graphomic", "pathgraphomic")):
        model = FlexibleFusion(opt, fdim=32, mmfdim=32 if "qbt" in opt.model else 64)
    else:
        raise NotImplementedError(f"Model {opt.model} not implemented")
    return model


def define_scheduler(opt, optimizer):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.lr_fix) / float(opt.n_epochs - opt.lr_fix)
        return lr_l

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_model(model, opt, preds):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt,
        },
        os.path.join(mkdir(f"{opt.ckpt_dir}"), f"{opt.model}_{opt.k}.pt"),
    )
    preds.to_csv(
        os.path.join(mkdir(f"{opt.ckpt_dir}/results"), f"{opt.model}_{opt.k}.csv")
    )


def make_results_table(metrics_list):
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


def log_epoch(
    epoch, model, train_loader, test_loader, loss_fn, opt, train_loss, all_preds
):
    _, train_acc, train_auc, c_train = evaluate(
        model, train_loader, loss_fn, opt, pd.DataFrame(all_preds)
    )
    test_loss, test_acc, test_auc, c_test = evaluate(model, test_loader, loss_fn, opt)
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
