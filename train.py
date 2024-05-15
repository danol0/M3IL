import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.datasets import pathologyDataset, select_min, get_splits
from src.networks import define_model
import numpy as np
from src.options import parse_args
from tabulate import tabulate
from src.utils import (
    loss_fn,
    save_model,
    evaluate,
    define_scheduler,
    make_results_table,
)

# from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(2019)
np.random.seed(2019)


def train(model, split_data, accelerator, opt, verbose=True):
    train_loader, test_loader = (
        DataLoader(
            pathologyDataset(split_data, s),
            batch_size=opt.batch_size,
            shuffle=True,
            collate_fn=select_min,
        )
        for s in ["train", "test"]
    )

    optim = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=opt.lr, weight_decay=opt.l2
    )
    scheduler = define_scheduler(opt, optim)
    model, train_loader, test_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, test_loader, optim, scheduler
    )

    pbar = tqdm(total=len(train_loader), position=0, leave=True, disable=not verbose)
    for epoch in range(opt.n_epochs):
        pbar.reset()
        model.train()

        if epoch == 10 and opt.model == "qbt":
            model.omic_net.freeze(False)

        for omics, path, event, time, grade, _ in train_loader:

            optim.zero_grad()
            _, grade_pred, surv_pred = model(x_omic=omics, x_path=path)
            loss = loss_fn(model, grade, time, event, grade_pred, surv_pred, opt)
            accelerator.backward(loss)
            optim.step()
            pbar.update(1)
            pbar.refresh()

        scheduler.step()
        train_loss, train_acc, _, c_train = evaluate(model, train_loader, opt)
        test_loss, test_acc, _, c_test = evaluate(model, test_loader, opt)

        desc = f"Epoch {epoch} (train/test) | Loss: {train_loss:.2f}/{test_loss:.2f} | "
        desc += f"Acc: {train_acc:.2f}/{test_acc:.2f} | " if opt.task != "surv" else ""
        desc += f"C-Index: {c_train:.2f}/{c_test:.2f} | " if opt.task != "grad" else ""
        pbar.set_description(desc)

    metrics = []
    for loader, name in [(train_loader, "Train"), (test_loader, "Test")]:
        metrics.append([name, *evaluate(model, loader, opt)])

    return model, metrics


if __name__ == "__main__":
    opt, str_opt = parse_args()
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    device = accelerator.device
    print(f"Device: {device}")
    opt.device = device

    split_data = get_splits(opt)
    metrics_list = []

    for i in range(1, opt.n_folds + 1):
        model = define_model(opt, i)
        (
            print(
                "Trainable parameters:",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
            if i == 1
            else None
        )
        model, metrics = train(model, split_data[i], accelerator, opt, verbose=True)

        mtable = [[f"Split {i}", "Loss", "Accuracy", "AUC", "C-Index"], *metrics]
        print(
            tabulate(
                mtable, headers="firstrow", tablefmt="rounded_grid", floatfmt=".3f"
            )
        )
        metrics_list.append(metrics[1][2:])
        save_model(model, opt, i)

    rtable = make_results_table(metrics_list, opt.n_folds)
    print(rtable)
    with open(f"checkpoints/{opt.task}/{opt.model}/results.txt", "w") as f:
        f.write(str_opt + "\n")
        f.write(rtable + "\n\n")
