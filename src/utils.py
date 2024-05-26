import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from tabulate import tabulate

import wandb

# from line_profiler import profile

# --- Training ---


# Predictions are not independent in Coxloss, ie calculating over batch != whole dataset
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


def print_load(ckpt):
    print(f"Loading {ckpt}")
    return torch.load(ckpt)


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


# --- Evaluation ---
def make_empty_data_dict():
    keys = [
        "patname",
        "grade_p_0",
        "grade_p_1",
        "grade_p_2",
        "hazard_pred",
        "grade",
        "event",
        "time",
    ]
    return {key: [] for key in keys}


@torch.no_grad()
def get_all_preds(model, data_loader):
    """Outputs dataframe of all predictions and ground truths."""

    model.eval()
    all_preds = make_empty_data_dict()

    for omics, path, graph, event, time, grade, patname in data_loader:

        _, grade_pred, hazard_pred = model(x_omic=omics, x_path=path, x_graph=graph)
        all_preds["patname"].extend(patname)

        for i in range(3):
            all_preds[f"grade_p_{i}"].extend(grade_pred[:, i].cpu().numpy())
        all_preds["hazard_pred"].extend(hazard_pred[:, 0].cpu().numpy())

        for key, value in {"grade": grade, "event": event, "time": time}.items():
            all_preds[key].extend(value.cpu().numpy())

    return pd.DataFrame(all_preds)


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, opt, precomp=None, return_preds=False):
    """Computes all metrics for a given model and dataset."""

    all_preds = get_all_preds(model, data_loader) if precomp is None else precomp

    # If instance level MIL, aggregate predictions by patient
    if opt.mil in ("instance", "paper"):
        # aggregrate by patient
        all_preds = all_preds.groupby("patname").agg(
            {
                "grade_p_0": "max",
                "grade_p_1": "max",
                "grade_p_2": "max",
                "hazard_pred": "mean",
                "grade": "first",
                "event": "first",
                "time": "first",
            }
        )

    grade_logits = all_preds[["grade_p_0", "grade_p_1", "grade_p_2"]].values

    accuracy = (
        (grade_logits.argmax(axis=1) == all_preds["grade"]).mean()
        if opt.task != "surv"
        else 0
    )
    auc = (
        roc_auc_score(pd.get_dummies(all_preds["grade"]), grade_logits, average="micro")
        if opt.task != "surv"
        else 0
    )
    c_indx = (
        concordance_index(
            all_preds["time"], -all_preds["hazard_pred"], all_preds["event"]
        )
        if opt.task != "grad"
        else 0
    )

    loss = 0
    if precomp is None:
        loss_inputs = [
            (
                torch.tensor(all_preds[key].values)
                if isinstance(key, str)
                else torch.tensor(key)
            )
            for key in ["grade", "time", "event", grade_logits, "hazard_pred"]
        ]
        loss = loss_fn(model, *loss_inputs).item()

    if return_preds:
        return loss, accuracy, auc, c_indx, all_preds
    else:
        return loss, accuracy, auc, c_indx
