import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.networks import MaxNet
from src.utils import get_splits, lambda_rule, loss_fn, calc_cindex
from src.data_loaders import omicDataset
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from tabulate import tabulate

# from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(0)
np.random.seed(0)


@torch.no_grad()
def test(model, test_loader):
    test_loss, test_acc, all_surv_test = 0, 0, []
    model.eval()
    for omics, censored, time, grade in test_loader:
        _, grade_pred, surv_pred = model(x_omic=omics)
        loss = loss_fn(model, grade, time, censored, grade_pred, surv_pred)
        test_acc += (grade_pred.argmax(dim=1) == grade).sum().item()
        test_loss += loss.item() * len(grade)
        all_surv_test.append(
            [surv_pred.cpu().numpy(), censored.cpu().numpy(), time.cpu().numpy()]
        )
    return test_loss, test_acc, all_surv_test


def train(n_epochs, model, split_data, verbose=True):
    accelerator = Accelerator(cpu=True)
    device = accelerator.device
    print(f"Device: {device}") if verbose else None

    train_loader, test_loader = (
        DataLoader(omicDataset(split_data, s), batch_size=64, shuffle=True)
        for s in ["train", "test"]
    )

    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.002)
    scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=lambda_rule)
    model, optim, train_loader, test_loader = accelerator.prepare(
        model, optim, train_loader, test_loader
    )

    train_len, test_len = len(train_loader.dataset), len(test_loader.dataset)
    pbar = tqdm(total=len(train_loader), position=0, leave=True, disable=not verbose)

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_acc, all_surv_train = 0, 0, []

        pbar.reset()
        for omics, censored, time, grade in train_loader:

            optim.zero_grad()
            _, grade_pred, surv_pred = model(x_omic=omics)
            loss = loss_fn(model, grade, time, censored, grade_pred, surv_pred)

            train_acc += (grade_pred.argmax(dim=1) == grade).sum().item()
            train_loss += loss.item() * len(grade)
            all_surv_train.append(
                [
                    surv_pred.detach().cpu().numpy(),
                    censored.cpu().numpy(),
                    time.cpu().numpy(),
                ]
            )

            accelerator.backward(loss)
            optim.step()
            pbar.update(1)

        scheduler.step()
        test_loss, test_acc, all_surv_test = test(model, test_loader)

        c_train, c_test = calc_cindex(all_surv_train), calc_cindex(all_surv_test)

        pbar.set_description(
            f"Epoch {epoch} | Loss: {train_loss/train_len:.2}/{test_loss/test_len:.2} | Acc: {train_acc/train_len:.2}/{test_acc/test_len:.2}, C-Index: {c_train:.2}/{c_test:.2}"
        )

    metrics = []
    for loader, name in [(train_loader, "Train"), (test_loader, "Test")]:
        loss, acc, all_surv = test(model, loader)
        c = calc_cindex(all_surv)
        metrics.append([name, loss / len(loader.dataset), acc / len(loader.dataset), c])

    return model, metrics


if __name__ == "__main__":

    split_data = get_splits(use_rnaseq=True)
    test_accs, test_c_idxs = [], []
    for i in range(1, 16):
        model = MaxNet(input_dim=320)
        model, metrics = train(50, model, split_data[i])
        print(
            tabulate(
                metrics,
                headers=[f"Split {i}", "Loss", "Accuracy", "C-Index"],
                floatfmt=".2f",
            )
        )
        test_accs.append(metrics[1][2])
        test_c_idxs.append(metrics[1][3])

    print(f"Mean test accuracy: {np.mean(test_accs):.2f}")
    print(f"Mean test c-index: {np.mean(test_c_idxs):.2f}")
