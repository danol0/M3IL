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


def calculate_metrics(predictions: pd.DataFrame, task: str) -> dict:
    """
    Calculates task relevant metrics for a given set of predictions.

    Args:
        predictions (pd.DataFrame): DataFrame of predictions.
        task (str): Task for which to calculate metrics.
    """
    assert predictions.index.is_unique

    grade_logits = predictions[["grade_p_0", "grade_p_1", "grade_p_2"]].values

    accuracy = (
        (grade_logits.argmax(axis=1) == predictions["grade"]).mean()
        if task != "surv"
        else 0
    )
    auc = (
        roc_auc_score(
            pd.get_dummies(predictions["grade"]), grade_logits, average="micro"
        )
        if task != "surv"
        else 0
    )
    c_indx = (
        concordance_index(
            predictions["time"], -predictions["hazard_pred"], predictions["event"]
        )
        if task != "grad"
        else 0
    )

    return {"accuracy": accuracy, "auc": auc, "c_indx": c_indx}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    opt: Namespace,
    precomp: pd.DataFrame = None,
    return_preds: bool = False,
) -> dict:
    """
    Evaluates a model against task relevant metrics.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation data.
        loss_fn (nn.Module): Loss function for evaluation.
        opt (Namespace): Command line arguments.
        precomp (pd.DataFrame): Precomputed predictions for evaluation.
        return_preds (bool): Whether to return a DataFrame of all predictions.
    """

    all_preds = get_all_preds(model, data_loader) if precomp is None else precomp

    # If instance level MIL, aggregate predictions by patient
    if opt.mil == "PFS":
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

    metrics = calculate_metrics(all_preds, opt.task)
    loss = 0
    if precomp is None:
        loss_inputs = [
            (torch.tensor(all_preds[key].values))
            for key in [
                "grade",
                "time",
                "event",
                ["grade_p_0", "grade_p_1", "grade_p_2"],
                "hazard_pred",
            ]
        ]
        loss = loss_fn(model, *loss_inputs)[1].item()

    metrics["loss"] = loss

    if return_preds:
        return metrics, all_preds
    else:
        return metrics
