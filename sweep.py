import os
from argparse import Namespace
from typing import Dict

import numpy as np
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
from train import train

def main():

    torch.manual_seed(2019)
    torch.use_deterministic_algorithms(True)
    np.random.seed(2019)

    opt = parse_args()
    opt.dry_run = 1
    opt.ckpt_dir = "checkpoints/sweep"
    opt.folds = 5
    opt.model = 'pathomic'
    opt.mil = 'global'
    wandb.init(config=opt)
    opt.l1 = wandb.config.l1
    # opt.l2 = wandb.config.l2
    opt.dropout = wandb.config.dropout
    opt.unfreeze_unimodal = wandb.config.unfreeze_unimodal
    opt.qbt_queries = wandb.config.qbt_queries
    opt.qbt_dim = wandb.config.qbt_dim
    opt.qbt = True
    opt.task = "grad"
    opt.n_epochs = 30
    cpu = True #if opt.model == "path" else True
    accelerator = Accelerator(cpu=cpu, step_scheduler_with_optimizer=False)
    opt.device = accelerator.device
    split_data = get_splits(opt, split_csv='sweep.csv')

    objective = "auc" if opt.task == "grad" else "c_indx"

    score = []
    for k in tqdm(range(1, len(split_data) + 1)):
        opt.k = k
        model = utils.define_model(opt)
        metrics = train(model, split_data[k], accelerator, opt, verbose=False)
        score.append(metrics["Test"][objective])

    wandb.log({"score": np.mean(score)})
    print(f"Mean {objective}: {np.mean(score)}")


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximise", "name": "score"},
    "parameters": {
        "l1": {"values": [0.0005]},
        # "l2": {"min": 1e-5, "max": 1e-2},
        "dropout": {"values": [0.4]},
        "unfreeze_unimodal": {"values": [10]},
        "qbt_queries": {"values": [16, 32, 48, 64]},
        "qbt_dim": {"values": [16, 32, 48, 64]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweep_mil")

wandb.agent(sweep_id, function=main, count=50, project="sweep_mil")
