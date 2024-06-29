import os
from argparse import Namespace
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import src.tools.train_utils as utils
from src.dataset.loaders import define_collate_fn, define_dataset
from src.dataset.preprocess import get_splits
from src.tools.evaluation import make_empty_data_dict, evaluate, calculate_metrics
from src.tools.options import parse_args


def CV_Main() -> None:
    """Cross-validates a model specified by command line args."""

    opt = parse_args()
    accelerator = utils.init_accelerator(opt)
    split_data = get_splits(opt)
    utils.assign_ckptdir_and_group(opt)
    print(opt)

    cv_metrics = []
    # --- Cross-validation Loop ---
    for k in range(1, opt.folds + 1):

        if opt.resume and os.path.exists(f"{opt.ckpt_dir}/{opt.model}_{k}.pt"):
            # Get metrics from previous run
            fold_preds = pd.read_csv(f"{opt.ckpt_dir}/results/{opt.model}_{k}.csv")
            mtest = calculate_metrics(
                fold_preds[fold_preds["split"] == "Test"], opt.task
            )
            cv_metrics.append([mtest["accuracy"], mtest["auc"], mtest["c_indx"]])
            print(f"Skipping split {k}")
            continue

        opt.k = k
        utils.set_seed(k)
        model = utils.define_model(opt)

        if k == 1:
            print(f"\n{model.n_params()} trainable parameters")

        with utils.configure_wandb(opt, k):
            wandb.watch(model)
            metrics = train(model, split_data[k], accelerator, opt)

        cv_metrics = utils.log_fold(cv_metrics, metrics, k)

    # --- End CV ---
    rtable = utils.make_cv_results_table(cv_metrics)
    print(rtable)

    if not opt.dry_run:
        with open(f"{opt.ckpt_dir}/results.txt", "w") as f:
            f.write(str(opt) + "\n")
            f.write(rtable + "\n\n")
        print(f"Results saved to ./{opt.ckpt_dir}/")


def train(
    model: nn.Module,
    split_data: Dict,
    accelerator: Accelerator,
    opt: Namespace,
    verbose: bool = True,
) -> Dict:
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
            define_dataset(opt)(split_data[split], opt),
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
        # Record all training predictions for efficient evaluation
        all_preds = make_empty_data_dict()

        if epoch == opt.unfreeze_unimodal:
            try:
                model.omic_net.freeze(False)
                model.graph_net.freeze(False)
            except AttributeError:
                pass

        for omic, path, graph, event, time, grade, patname in train_loader:

            # --- Training Step ---
            optim.zero_grad()
            _, grade_pred, hazard_pred = model(x_omic=omic, x_path=path, x_graph=graph)
            # Objective includes reg term, loss does not and is used for logging
            objective, loss = loss_fn(
                model, grade, time, event, grade_pred, hazard_pred
            )
            accelerator.backward(objective)
            optim.step()

            # --- Logging ---
            if verbose:
                train_loss += loss.item() * len(patname)
                all_preds["patname"].extend(patname)
                for i in range(3):
                    all_preds[f"grade_p_{i}"].extend(
                        grade_pred[:, i].detach().cpu().numpy()
                    )
                all_preds["hazard_pred"].extend(
                    hazard_pred[:, 0].detach().cpu().numpy()
                )
                for key, value in {
                    "grade": grade,
                    "event": event,
                    "time": time,
                }.items():
                    all_preds[key].extend(value.cpu().numpy())

        # --- End of epoch ---
        scheduler.step()
        if verbose:
            train_loss /= samples
            desc = utils.log_epoch(
                opt,
                model,
                epoch,
                train_loader,
                test_loader,
                loss_fn,
                train_loss,
                all_preds,
            )
            pbar.set_description(desc)

    # --- Evaluation ---
    all_metrics, all_preds = {}, []
    for loader, split in [(train_loader, "Train"), (test_loader, "Test")]:
        split_metrics, split_preds = evaluate(
            model, loader, loss_fn, opt, return_preds=True
        )
        split_preds["split"] = split
        all_preds.append(split_preds)
        all_metrics[split] = split_metrics
    all_preds = pd.concat(all_preds)
    utils.save_model(model, opt, all_preds) if not opt.dry_run else None

    # --- Cleanup ---
    model, train_loader, test_loader, optim, scheduler = accelerator.clear(
        model, train_loader, test_loader, optim, scheduler
    )

    return all_metrics


if __name__ == "__main__":
    CV_Main()
