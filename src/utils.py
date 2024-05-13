import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from src.networks import MaxNet, QBTNet
import pickle
from collections import defaultdict


def define_model(opt, k):
    omic_dim = 320 if opt.use_rna else 80
    if opt.model == "omic":
        model = MaxNet(input_dim=omic_dim, omic_dim=32, dropout=opt.dropout)
    if opt.model == "qbt":
        model = QBTNet(
            k,
            feature_dim=32,
            n_queries=16,
            batch_size=opt.batch_size,
            transformer_layers=12,
            dropout=opt.dropout,
            omic_xdim=omic_dim,
        )
    else:
        raise NotImplementedError(f"Model {opt.model} not implemented")
    return model


@torch.no_grad()
def evaluate(model, data_loader, opt):
    all_grade, all_surv, loss = [], [], 0
    model.eval()
    for omics, path, censored, time, grade in data_loader:
        _, grade_pred, surv_pred = model(x_omic=omics, x_path=path)
        loss += loss_fn(
            model, grade, time, censored, grade_pred, surv_pred, opt
        ).item() * len(grade)
        grade_onehot = np.eye(3)[grade.cpu().numpy()]
        all_grade.append([grade_pred.cpu().numpy(), grade_onehot])
        all_surv.append(
            [surv_pred.cpu().numpy(), censored.cpu().numpy(), time.cpu().numpy()]
        )
    acc = (
        np.vstack([x[0] for x in all_grade]).argmax(axis=1)
        == np.vstack([x[1] for x in all_grade]).argmax(axis=1)
    ).mean()
    auc = roc_auc_score(
        np.concatenate([x[1] for x in all_grade]),
        np.concatenate([x[0] for x in all_grade]),
        average="micro",
    )
    c = calc_cindex(all_surv)
    loss /= len(data_loader.dataset)
    return loss, acc, auc, c


def calc_cindex(all_surv):
    all_preds = np.concatenate([x[0] for x in all_surv])
    all_censor = np.concatenate([x[1] for x in all_surv])
    all_time = np.concatenate([x[2] for x in all_surv])
    return concordance_index(all_time, -all_preds, all_censor)


def loss_fn(model, grade, time, censored, grade_pred, surv_pred, opt):
    w_nll = 0 if opt.task == "surv" else 1
    w_cox = 0 if opt.task == "grad" else 1
    return (
        w_nll * F.nll_loss(grade_pred, grade)
        + opt.w_reg * l1_reg(model)
        + w_cox * CoxLoss(time, censored, surv_pred)
    )


def lambda_rule(epoch, epoch_count=0, niter=0, niter_decay=50):
    lr_l = 1.0 - max(0, epoch + epoch_count - niter) / float(niter_decay + 1)
    return lr_l


def l1_reg(model):
    return sum(torch.abs(W).sum() for W in model.parameters())


def CoxLoss(survtime, censor, hazard_pred):
    R_mat = (survtime.repeat(len(survtime), 1) >= survtime.unsqueeze(1)).int()
    theta = hazard_pred.view(-1)
    exp_theta = theta.exp()
    loss_cox = -torch.mean((theta - (exp_theta * R_mat).sum(dim=1).log()) * censor)
    return loss_cox


def get_splits(opt, data_dir="./data"):
    metadata, dataset = getCleanAllDataset(use_rnaseq=opt.use_rna)
    split_data = {}

    pnas_splits = pd.read_csv(f"{data_dir}/splits/pnas_splits.csv")
    pnas_splits.columns = ["TCGA ID"] + [k for k in range(1, 16)]
    pnas_splits.set_index("TCGA ID", inplace=True)
    pnas_splits = pnas_splits.map(lambda x: x.lower())

    vgg_ftype = "surv" if opt.task == "multi" else opt.task
    vgg_feats = pickle.load(open(f"{data_dir}/vgg_features_{vgg_ftype}.pkl", "rb"))
    pat2img = defaultdict(list)
    for img_fname in vgg_feats.keys():
        pat2img[img_fname[:12]].append(img_fname)

    for k in range(1, opt.n_folds + 1):
        split_data[k] = {
            split: {
                pat: {
                    "x_omic": dataset.loc[pat].drop(metadata).values.astype(np.float32),
                    "x_path": np.stack(
                        [vgg_feats[img_fname] for img_fname in pat2img[pat]]
                    ),
                    "y": [
                        dataset.loc[pat]["censored"],
                        dataset.loc[pat]["Survival months"],
                        dataset.loc[pat]["Grade"],
                    ],
                }
                for pat in pnas_splits.index[
                    (pnas_splits[k] == split) & (pnas_splits.index.isin(dataset.index))
                ]
            }
            for split in ["train", "test"]
        }

        # Standardise omics
        all_train_omics = np.vstack(
            [split_data[k]["train"][pat]["x_omic"] for pat in split_data[k]["train"]]
        )
        scaler = StandardScaler().fit(all_train_omics)
        for split in ["train", "test"]:
            for pat in split_data[k][split]:
                split_data[k][split][pat]["x_omic"] = scaler.transform(
                    split_data[k][split][pat]["x_omic"].reshape(1, -1)
                ).flatten()

    return split_data


def getCleanAllDataset(data_dir="./data/omics", use_rnaseq=False):
    metadata = [
        "Histology",
        "Grade",
        "Molecular subtype",
        "censored",
        "Survival months",
    ]

    all_dataset = pd.read_csv(f"{data_dir}/all_dataset.csv").drop("indexes", axis=1)
    all_dataset.set_index("TCGA ID", inplace=True)

    all_grade = pd.read_csv(f"{data_dir}/grade_data.csv")
    all_grade["Histology"] = all_grade["Histology"].str.replace(
        "astrocytoma (glioblastoma)", "glioblastoma", regex=False
    )
    all_grade.set_index("TCGA ID", inplace=True)

    assert pd.Series(all_dataset.index).equals(pd.Series(sorted(all_grade.index)))
    all_dataset = all_dataset.join(
        all_grade[["Histology", "Grade", "Molecular subtype"]],
        how="inner",
        on="TCGA ID",
    )

    all_dataset["Grade"] = all_dataset["Grade"] - 2
    all_dataset["censored"] = 1 - all_dataset["censored"]

    if use_rnaseq:
        gbm = pd.read_csv(
            f"{data_dir}/mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt",
            sep="\t",
            skiprows=1,
            index_col=0,
        )
        lgg = pd.read_csv(
            f"{data_dir}/mRNA_Expression_Zscores_RSEM.txt",
            sep="\t",
            skiprows=1,
            index_col=0,
        )
        gbm = gbm[gbm.columns[~gbm.isnull().all()]]
        lgg = lgg[lgg.columns[~lgg.isnull().all()]]
        glioma_RNAseq = gbm.join(lgg, how="inner").T
        glioma_RNAseq = glioma_RNAseq.dropna(axis=1)
        glioma_RNAseq.columns = [gene + "_rnaseq" for gene in glioma_RNAseq.columns]
        glioma_RNAseq.index = [patname[:12] for patname in glioma_RNAseq.index]
        # keep first occurence of duplicated index
        glioma_RNAseq = glioma_RNAseq.iloc[~glioma_RNAseq.index.duplicated()]
        glioma_RNAseq.index.name = "TCGA ID"
        all_dataset = all_dataset.join(glioma_RNAseq, how="inner")

    # Drop patients with missing data
    pat_missing_moltype = all_dataset[all_dataset["Molecular subtype"].isna()].index
    pat_missing_idh = all_dataset[all_dataset["idh mutation"].isna()].index
    pat_missing_1p19q = all_dataset[all_dataset["codeletion"].isna()].index
    assert pat_missing_moltype.equals(pat_missing_idh)
    assert pat_missing_moltype.equals(pat_missing_1p19q)
    all_dataset = all_dataset.drop(pat_missing_moltype)
    pat_missing_grade = all_dataset[all_dataset["Grade"].isna()].index
    pat_missing_histype = all_dataset[all_dataset["Histology"].isna()].index
    assert pat_missing_histype.equals(pat_missing_grade)
    all_dataset = all_dataset.drop(pat_missing_histype)

    # 3 patients have no omics data: we remove them
    missing = all_dataset.drop(metadata, axis=1).isna().any(axis=1)
    print(f"Removing {missing.sum()} patients with missing omics data")
    all_dataset = all_dataset[~missing]

    return metadata, all_dataset
