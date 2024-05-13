import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import get_splits, lambda_rule, loss_fn, calc_cindex, evaluate, define_model
from src.data_loaders import omicDataset
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from tabulate import tabulate
from src.options import parse_args

# from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(2019)
np.random.seed(2019)


def train(model, split_data, accelerator, opt, verbose=True):
    drop_last = True if opt.model == 'qbt' else False
    train_loader, test_loader = (
        DataLoader(omicDataset(split_data, s), batch_size=opt.batch_size, shuffle=True, drop_last=drop_last)
        for s in ["train", "test"]
    )

    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=opt.lr)
    scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=lambda_rule)
    model, train_loader, test_loader, optim, scheduler = accelerator.prepare(
        model, train_loader, test_loader, optim, scheduler
    )

    pbar = tqdm(total=len(train_loader), position=0, leave=True, disable=not verbose)
    for epoch in range(opt.n_epochs):
        pbar.reset()
        model.train()
        train_loss, train_acc, all_surv_train = 0, 0, []
        for omics, path, censored, time, grade in train_loader:

            optim.zero_grad()
            _, grade_pred, surv_pred = model(x_omic=omics, x_path=path)
            loss = loss_fn(model, grade, time, censored, grade_pred, surv_pred, opt)
            accelerator.backward(loss)
            optim.step()

            train_acc += (grade_pred.argmax(dim=1) == grade).sum().item()
            train_loss += loss.item() * len(grade)
            all_surv_train.append(
                [
                    surv_pred.detach().cpu().numpy(),
                    censored.cpu().numpy(),
                    time.cpu().numpy(),
                ]
            )

            pbar.update(1)
            pbar.refresh()

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        c_train = calc_cindex(all_surv_train)
        test_loss, test_acc, _, test_cindx = evaluate(model, test_loader, opt)

        pbar.set_description(
            f"Epoch {epoch} | Loss: {train_loss:.2f}/{test_loss:.2f} | Acc: {train_acc:.2f}/{test_acc:.2f}, C-Index: {c_train:.2f}/{test_cindx:.2f}"
        )
        pbar.refresh()

    metrics = []
    for loader, name in [(train_loader, "Train"), (test_loader, "Test")]:
        metrics.append([name, *evaluate(model, loader, opt)])

    return model, metrics


if __name__ == "__main__":
    opt = parse_args()
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    device = accelerator.device
    print(f"Device: {device}")

    split_data = get_splits(opt)
    metrics_list = []

    for i in range(1, opt.n_folds + 1):
        model = define_model(opt, i)
        model, metrics = train(model, split_data[i], accelerator, opt, verbose=True)
        print(
            "\n",
            tabulate(
                metrics,
                headers=[f"Split {i}", "Loss", "Accuracy", "AUC", "C-Index"],
            ),
        )

        metrics_list.append(metrics[1][2:])

    mean_std_metrics = [
        ['Mean', *np.mean(metrics_list, axis=0)],
        ['Std', *np.std(metrics_list, axis=0)]
    ]

    print(
        "\n",
        tabulate(
            mean_std_metrics,
            headers=["Accuracy", "AUC", "C-Index"],
        ),
    )
