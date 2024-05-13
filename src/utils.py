import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from lifelines.utils import concordance_index


def calc_cindex(all_surv):
    all_preds = np.concatenate([x[0] for x in all_surv])
    all_censor = np.concatenate([x[1] for x in all_surv])
    all_time = np.concatenate([x[2] for x in all_surv])
    return concordance_index(all_time, -all_preds, all_censor)


def loss_fn(model, grade, time, censored, grade_pred, surv_pred):
    return F.nll_loss(grade_pred, grade) + 3e-4 * l1_reg(model) + CoxLoss(time, censored, surv_pred)


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


def get_splits(data_dir="./data", use_rnaseq=False):
    metadata, dataset = getCleanAllDataset(use_rnaseq=use_rnaseq)
    split_data = {}

    pnas_splits = pd.read_csv(f"{data_dir}/splits/pnas_splits.csv")
    pnas_splits.columns = ["TCGA ID"] + [k for k in range(1, 16)]
    pnas_splits.set_index("TCGA ID", inplace=True)
    pnas_splits = pnas_splits.map(lambda x: x.lower())

    for k in range(1, 16):
        split_data[k] = {
            split: {
                pat: {
                    "x_omic": dataset.loc[pat].drop(metadata).values.astype(np.float32),
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
