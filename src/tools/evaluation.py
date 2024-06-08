from argparse import Namespace

import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score


# --- Evaluation ---
def make_empty_data_dict() -> dict:
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
def get_all_preds(
    model: nn.Module, data_loader: torch.utils.data.DataLoader
) -> pd.DataFrame:
    """Creates dataframe of all predictions and ground truths for model evaluation."""

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
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    opt: Namespace,
    precomp: pd.DataFrame = None,
    return_preds: bool = False,
) -> tuple:
    """
    Evaluates a model against task relevant metrics.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation data.
        loss_fn (nn.Module): Loss function for evaluation.
        opt (Namespace): Command line arguments.
        precomp (pd.DataFrame): Precomputed predictions for evaluation.
        return_preds (bool): Whether to return the predictions DataFrame.
    """

    all_preds = get_all_preds(model, data_loader) if precomp is None else precomp

    # If instance level MIL, aggregate predictions by patient
    if opt.mil == "PFS":
        # Aggregrate by patient
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

    assert all_preds.index.is_unique

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
