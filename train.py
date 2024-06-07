import os
from argparse import Namespace
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.tools.train_utils as utils
import wandb
from src.dataset.loaders import define_collate_fn, define_dataset
from src.dataset.preprocess import get_splits
from src.tools.evaluation import make_empty_data_dict
from src.tools.options import parse_args

# from line_profiler import profile
# from torch.profiler import profile, record_function, ProfilerActivity


def CV_Main() -> None:
    """Cross-validates a model specified by command line args."""

    opt, str_opt = parse_args()
    accelerator = init_environment(opt)
    split_data = get_splits(opt)
    assign_ckptdir_and_group(opt)

    metrics_list = []
    # --- Cross-validation Loop ---
    for k in range(1, opt.folds + 1):

        if opt.resume and os.path.exists(f"{opt.ckpt_dir}/{opt.model}_{k}.pt"):
            print(f"Skipping split {k}")
            continue

        opt.k = k
        model = utils.define_model(opt)

        if k == 1:
            print(model, f"\n{model.n_params()} trainable parameters")

        with configure_wandb(opt, k):
            model, metrics, preds = train(model, split_data[k], accelerator, opt)

        metrics_list = log_fold(metrics_list, metrics, k)
        utils.save_model(model, opt, preds) if not opt.dry_run else None
        accelerator.free_memory()

    # --- End CV ---
    rtable = utils.make_results_table(metrics_list)
    print(rtable)

    if not opt.dry_run:
        with open(f"{opt.ckpt_dir}/results.txt", "w") as f:
            f.write(str_opt + "\n")
            f.write(rtable + "\n\n")
        print(f"Results saved to ./{opt.ckpt_dir}/")


def train(
    model: nn.Module,
    split_data: Dict,
    accelerator: Accelerator,
    opt: Namespace,
    verbose: bool = True,
):
    """
    Train a model for a single split.

    Args:
        model (nn.Module): Model to train
        split_data (Dict): Train and test data for a given split
        accelerator (Accelerator): Accelerator object
        opt (Namespace): Command line arguments
        verbose (bool): Compute and log metrics every epoch (performance cost)
    """
    # --- Initialisation ---
    train_loader, test_loader = (
        DataLoader(
            define_dataset(opt)(split_data, split, opt),
            batch_size=opt.batch_size,
            shuffle=True,
            collate_fn=define_collate_fn(opt),
        )
        for split in ["train", "test"]
    )
    samples = len(train_loader.dataset)
    (
        print(
            f"{samples} train samples | {len(train_loader) * opt.n_epochs} gradient steps"
        )
        if verbose
        else None
    )
    optim = torch.optim.Adam(
        model.parameters(), betas=(opt.adam_b1, 0.999), lr=opt.lr, weight_decay=opt.l2
    )
    loss_fn = utils.PathomicLoss(opt)
    scheduler = utils.define_scheduler(opt, optim)
    model, train_loader, test_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, test_loader, optim, scheduler
    )

    # --- Training loop ---
    pbar = tqdm(range(opt.n_epochs), total=opt.n_epochs, disable=not verbose)
    for epoch in pbar:
        model.train()
        train_loss = 0
        # We record all training predictions for efficient evaluation
        all_preds = make_empty_data_dict()

        if epoch == opt.unfreeze_unimodal:
            if "pathomic" in opt.model:
                pbar.set_description("Unfreezing omic")
                model.omic_net.freeze(False)
            if "graphomic" in opt.model:
                pbar.set_description("Unfreezing omic and graph")
                model.omic_net.freeze(False)
                model.graph_net.freeze(False)

        for omic, path, graph, event, time, grade, patname in train_loader:

            # --- Training Step ---
            optim.zero_grad()
            _, grade_pred, hazard_pred = model(x_omic=omic, x_path=path, x_graph=graph)
            loss = loss_fn(model, grade, time, event, grade_pred, hazard_pred)
            accelerator.backward(loss)
            optim.step()

            # --- Logging ---
            train_loss += loss.item() * len(patname)
            if not verbose:
                continue
            all_preds["patname"].extend(patname)
            for i in range(3):
                all_preds[f"grade_p_{i}"].extend(
                    grade_pred[:, i].detach().cpu().numpy()
                )
            all_preds["hazard_pred"].extend(hazard_pred[:, 0].detach().cpu().numpy())
            for key, value in {"grade": grade, "event": event, "time": time}.items():
                all_preds[key].extend(value.cpu().numpy())

        # --- End of epoch ---
        scheduler.step()
        if not verbose:
            continue
        train_loss /= samples
        desc = utils.log_epoch(
            epoch, model, train_loader, test_loader, loss_fn, opt, train_loss, all_preds
        )
        pbar.set_description(desc)

    # --- Evaluation ---
    metrics, preds = {}, []
    for loader, split in [(train_loader, "Train"), (test_loader, "Test")]:
        loss, accuracy, auc, c_indx, all_preds = utils.evaluate(
            model, loader, loss_fn, opt, return_preds=True
        )
        all_preds["split"] = split
        preds.append(all_preds)
        metrics[split] = {
            "loss": loss,
            "accuracy": accuracy,
            "auc": auc,
            "c_indx": c_indx,
        }
    # Dataframe of all train/test predictions
    all_preds = pd.concat(preds)

    # --- Cleanup ---
    model, train_loader, test_loader, optim, scheduler = accelerator.clear(
        model, train_loader, test_loader, optim, scheduler
    )

    return model, metrics, all_preds


def init_environment(opt: Namespace) -> Accelerator:
    """Initialises global run settings."""

    utils.set_seed(2019)
    os.environ["WANDB_SILENT"] = "true"
    cpu = False if opt.model == "path" else True
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    opt.device = accelerator.device
    print(f"Device: {opt.device}")
    return accelerator


def configure_wandb(opt: Namespace, k: int) -> wandb.run:
    """Initialise a WandB run that tracks a single fold as part of a CV group."""

    return wandb.init(
        reinit=True,
        mode="disabled" if opt.dry_run else "online",
        project="mmmil",
        config=opt,
        group=opt.group,
        name=f"{opt.group}_{k}",
    )


def assign_ckptdir_and_group(opt: Namespace) -> None:
    """Assigns checkpoint directory and CV group."""

    rna = "_rna" if (opt.rna and "omic" in opt.model) else ""
    attn = "_attn" if opt.attn_pool else ""
    # Ignore MIL for omic as there is only 1 instance per patient
    if opt.model == "omic":
        opt.ckpt_dir = f"checkpoints/{opt.task}/{opt.model}{rna}"
        opt.group = f"{opt.task}_{opt.model}{rna}"
    else:
        opt.ckpt_dir = f"checkpoints/{opt.task}/{opt.model}{rna}_{opt.mil}{attn}"
        opt.group = f"{opt.task}_{opt.model}{rna}_{opt.mil}{attn}"
    print(f"Checkpoint dir: ./{opt.ckpt_dir}/")


def log_fold(metrics_list: list, metrics: Dict, k: int) -> list:
    """Log metrics for a single fold and return updated metrics list."""

    fold_table = [[f"Split {k}", "Loss", "Accuracy", "AUC", "C-Index"]]
    for split, m in metrics.items():
        fold_table.append([split, m["loss"], m["accuracy"], m["auc"], m["c_indx"]])
    tkwgs = {"headers": "firstrow", "tablefmt": "rounded_grid", "floatfmt": ".3f"}
    print(tabulate(fold_table, **tkwgs))

    mtest = metrics["Test"]
    metrics_list.append([mtest["accuracy"], mtest["auc"], mtest["c_indx"]])
    return metrics_list


if __name__ == "__main__":
    CV_Main()
