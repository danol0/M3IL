import os

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.utils as utils
import wandb
from src.datasets import define_collate_fn, define_dataset, get_splits
from src.networks import define_model
from src.options import parse_args

# from line_profiler import profile
# from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(2019)
np.random.seed(2019)


def train(model, split_data, accelerator, opt):
    train_loader, test_loader = (
        DataLoader(
            define_dataset(opt)(split_data, split, opt),
            batch_size=opt.batch_size,
            shuffle=True,
            collate_fn=define_collate_fn(opt),
        )
        for split in ["train", "test"]
    )
    print(
        f"{len(train_loader.dataset)} train samples | {len(train_loader) * opt.n_epochs} gradient steps"
    )

    optim = torch.optim.Adam(
        model.parameters(), betas=(opt.adam_b1, 0.999), lr=opt.lr, weight_decay=opt.l2
    )
    scheduler = utils.define_scheduler(opt, optim)
    model, train_loader, test_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, test_loader, optim, scheduler
    )
    loss_fn = utils.PathomicLoss(opt)

    pbar = tqdm(range(opt.n_epochs), total=opt.n_epochs)
    for epoch in pbar:

        model.train()
        train_loss = 0
        all_preds = utils.make_empty_data_dict()

        if epoch == 5:
            if opt.model in ("pathomic_qbt", "pathomic"):
                model.omic_net.freeze(False)
            if opt.model in ("graphomic", "pathgraphomic"):
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
            all_preds["patname"].extend(patname)
            for i in range(3):
                all_preds[f"grade_p_{i}"].extend(
                    grade_pred[:, i].detach().cpu().numpy()
                )
            all_preds["hazard_pred"].extend(hazard_pred[:, 0].detach().cpu().numpy())
            for key, value in {"grade": grade, "event": event, "time": time}.items():
                all_preds[key].extend(value.cpu().numpy())

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        desc = utils.log_epoch(
            epoch, model, train_loader, test_loader, loss_fn, opt, train_loss, all_preds
        )
        pbar.set_description(desc)

    metrics, preds = [], []
    for loader, name in [(train_loader, "Train"), (test_loader, "Test")]:
        loss, accuracy, auc, c_indx, all_preds = utils.evaluate(
            model, loader, loss_fn, opt, return_preds=True
        )
        all_preds["split"] = name
        preds.append(all_preds)
        metrics.append([name, loss, accuracy, auc, c_indx])
    all_preds = pd.concat(preds)

    return model, metrics, all_preds


if __name__ == "__main__":
    opt, str_opt = parse_args()
    print(str_opt)
    cpu = False if opt.model == "path" else True
    accelerator = Accelerator(cpu=cpu, step_scheduler_with_optimizer=False)
    device = accelerator.device
    print(f"Device: {device}")
    opt.device = device
    os.environ["WANDB_SILENT"] = "true"

    split_data = get_splits(opt)
    metrics_list = []
    rna = "_rna" if (opt.rna and "omic" in opt.model) else ""
    attn = "_attn" if opt.attn_pool else ""
    # Ignore MIL for omic as there is only 1 instance per patient
    if opt.model == "omic":
        opt.ckpt_dir = f"checkpoints/{opt.task}/{opt.model}{rna}"
        group = f"{opt.task}_{opt.model}{rna}"
    else:
        opt.ckpt_dir = f"checkpoints/{opt.task}/{opt.model}{rna}_{opt.mil}{attn}"
        group = f"{opt.task}_{opt.model}{rna}_{opt.mil}{attn}"
    print(f"Checkpoint dir: ./{opt.ckpt_dir}/")
    for k in range(1, opt.folds + 1):
        if opt.resume and os.path.exists(f"{opt.ckpt_dir}/{opt.model}_{k}.pt"):
            print(f"Skipping split {k}")
            continue
        opt.k = k
        wandb.init(
            reinit=True,
            mode="disabled" if opt.dry_run else "online",
            project="mmmil",
            config=opt,
            group=group,
            name=f"{group}_{k}",
        )

        model = define_model(opt)
        if k == 1:
            print(model)
            print(
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} params"
            )
        model, metrics, preds = train(model, split_data[k], accelerator, opt)

        mtable = [[f"Split {k}", "Loss", "Accuracy", "AUC", "C-Index"], *metrics]
        tkwgs = {"headers": "firstrow", "tablefmt": "rounded_grid", "floatfmt": ".3f"}
        print(tabulate(mtable, **tkwgs))

        metrics_list.append(metrics[1][2:])
        utils.save_model(model, opt, preds) if not opt.dry_run else None

    rtable = utils.make_results_table(metrics_list)
    print(rtable)

    if not opt.dry_run:
        with open(f"{opt.ckpt_dir}/results.txt", "w") as f:
            f.write(str_opt + "\n")
            f.write(rtable + "\n\n")
        print(f"Results saved to ./{opt.ckpt_dir}/")
