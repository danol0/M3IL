import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
import torch.optim.lr_scheduler as lr_scheduler
import os
from tabulate import tabulate


# --- Utility ---


def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_model(model, opt, k):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt,
            # 'metrics' : metrics
        },
        os.path.join(
            mkdir(f"checkpoints/{opt.task}/{opt.model}"), f"{opt.model}_{k}.pt"
        ),
    )


def make_results_table(metrics_list, n_folds):
    df = pd.DataFrame(
        metrics_list,
        columns=["Accuracy", "AUC", "C-Index"],
        index=range(1, n_folds + 1),
    )
    df.index.name = "Fold"
    df = df.loc[:, (df != 0).any(axis=0)]
    df.loc["mean"] = df.mean()
    df.loc["std"] = df.std()
    return tabulate(df.T, headers="keys", tablefmt="rounded_grid", floatfmt=".3f")


# --- Evaluation ---
@torch.no_grad()
def get_all_preds(model, data_loader):
    """Outputs dictionary of all predictions and ground truths."""

    model.eval()
    keys = ["patname", "grade_pred", "surv_pred", "grade", "event", "time"]
    all_preds = {key: [] for key in keys}

    for omics, path, event, time, grade, patname in data_loader:

        all_preds["patname"].extend(patname)  # Handled separately as str

        data = [*model(x_omic=omics, x_path=path)[1:], grade, event, time]
        for key, value in zip(keys[1:], data):
            all_preds[key].extend(value.cpu().numpy())
    # Output is a dictionary of numpy arrays
    return {key: np.array(value) for key, value in all_preds.items()}


@torch.no_grad()
def evaluate(model, data_loader, opt):
    """Computes all metrics for a given model and dataset."""

    all_preds = get_all_preds(model, data_loader)
    accuracy = (
        (all_preds["grade_pred"].argmax(axis=1) == all_preds["grade"]).mean()
        if opt.task != "surv"
        else 0
    )
    grade_onehot = np.eye(3)[all_preds["grade"]]  # n x 3
    auc = (
        roc_auc_score(grade_onehot, all_preds["grade_pred"], average="micro")
        if opt.task != "surv"
        else 0
    )
    c_indx = (
        concordance_index(
            all_preds["time"], -all_preds["surv_pred"], all_preds["event"]
        )
        if opt.task != "grad"
        else 0
    )

    loss_inputs = [
        torch.tensor(all_preds[key])
        for key in ["grade", "time", "event", "grade_pred", "surv_pred"]
    ]
    loss = loss_fn(model, *loss_inputs, opt).item()
    return loss, accuracy, auc, c_indx


def calc_cindex(all_surv):
    """all_surv: list of [surv_pred, event, time] for each batch."""
    all_preds = np.concatenate([x[0] for x in all_surv])
    all_event = np.concatenate([x[1] for x in all_surv])
    all_time = np.concatenate([x[2] for x in all_surv])
    return concordance_index(all_time, -all_preds, all_event)


# --- Training ---


def loss_fn(model, grade, time, event, grade_pred, surv_pred, opt):
    w_nll = 0 if opt.task == "surv" else 1
    w_cox = 0 if opt.task == "grad" else 1
    return (
        w_nll * F.nll_loss(grade_pred, grade)
        + opt.l1 * l1_reg(model)
        + w_cox * CoxLoss(time, event, surv_pred)
    )


def CoxLoss(survtime, event, hazard_pred):
    R_mat = (survtime.repeat(len(survtime), 1) >= survtime.unsqueeze(1)).int()
    theta = hazard_pred.view(-1)
    exp_theta = theta.exp()
    loss_cox = -torch.mean((theta - (exp_theta * R_mat).sum(dim=1).log()) * event)
    return loss_cox


def l1_reg(model):
    return sum(torch.abs(W).sum() for W in model.parameters())


def define_scheduler(opt, optimizer):
    def lambda_rule(epoch):
        lr_l = 1.0 - epoch / float(opt.n_epochs)
        return lr_l

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
