import os
import pickle
from argparse import Namespace
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_splits(opt: Namespace, split_csv='pnas_splits.csv') -> dict:
    """
    Constructs the data splits for the TCGA-GBMLGG dataset.
    Only includes data necessary for the specified model.
    Standardises omics data.

    Returns a nested dictionary of the form:
        split {
            patient_id {
                x_omic: omic data tensor
                x_path: pre-extracted histology feature tensor or list of image paths
                x_graph: list of graph data paths
                y: list of event, time, grade labels
            }
        }

    Args:
        opt (Namespace): Command line arguments
        split_csv (pd.DataFrame): Name of csv file containing splits.
    """

    data_dir = opt.data_dir
    labels, dataset = get_all_dataset(
        use_rnaseq=opt.rna,
        rm_missing_omics=1 if "omic" in opt.model else 0,
        rm_missing_grade=1 if opt.task in ["multi", "grad"] else 0,
        data_dir=data_dir,
    )
    split_data = {}

    splits = pd.read_csv(f"{data_dir}/splits/{split_csv}")
    splits.columns = ["TCGA ID"] + [k for k in range(1, splits.shape[1])]
    splits.set_index("TCGA ID", inplace=True)
    splits = splits.applymap(lambda x: x.lower())
    splits.to_csv(f"{data_dir}/splits/test.csv", index=True)
    assert opt.folds <= splits.shape[1], "Number of folds exceeds number of splits in CSV"

    if opt.pre_encoded_path and "path" in opt.model:
        if opt.use_vggnet:
            vgg_ftype = "surv" if opt.task == "multi" else opt.task
            try:
                vgg_feats = pickle.load(
                    open(f"{data_dir}/path/vgg_features_{vgg_ftype}.pkl", "rb")
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Pre-extracted VGG features not found for {vgg_ftype}"
                )
        else:
            try:
                vgg_feats = pickle.load(
                    open(f"{data_dir}/path/resnet_features.pkl", "rb")
                )
            except FileNotFoundError:
                raise FileNotFoundError("Pre-extracted ResNet features not found")

    pat2roi = defaultdict(list)
    roi2patch = defaultdict(list)
    if "graph" in opt.model or "path" in opt.model:
        for roi_fname in os.listdir(f"{data_dir}/graphs/"):
            pat2roi[roi_fname[:12]].append(roi_fname.rstrip(".pt"))
            # We do this with nested loops to preserve the order of the patches
            # This allows us to match the patches to parent graphs as done in the paper
            if opt.pre_encoded_path and "path" in opt.model:
                path_dict = vgg_feats["1"] if opt.use_vggnet else vgg_feats
                for img_fname in path_dict.keys():
                    if img_fname.startswith(roi_fname.rstrip(".pt")):
                        roi2patch[roi_fname.rstrip(".pt")].append(img_fname)

    for k in range(1, opt.folds + 1):
        split_data[k] = {
            split: {
                pat: {
                    "x_omic": (
                        dataset.loc[pat].drop(labels).values.astype(np.float32)
                        if "omic" in opt.model
                        else []
                    ),
                    "x_path": (
                        (
                            np.stack(
                                [
                                    (
                                        vgg_feats[str(k)][patch]
                                        if opt.use_vggnet
                                        else vgg_feats[patch]
                                    )
                                    for roi in pat2roi[pat]
                                    for patch in roi2patch[roi]
                                ]
                            )
                            if opt.pre_encoded_path
                            else [
                                f"{data_dir}/path/ROIs/{roi}.png"
                                for roi in pat2roi[pat]
                            ]
                        )
                        if "path" in opt.model
                        else []
                    ),
                    "x_graph": (
                        [f"{data_dir}/graphs/{roi}.pt" for roi in pat2roi[pat]]
                        if "graph" in opt.model
                        else []
                    ),
                    "y": [
                        dataset.loc[pat]["Event"],
                        dataset.loc[pat]["Survival months"],
                        dataset.loc[pat]["Grade"],
                    ],
                }
                for pat in splits.index[
                    (splits[k] == split) & (splits.index.isin(dataset.index))
                ]
            }
            for split in ["train", "test"]
        }

        # Standardise omics
        if "omic" in opt.model:
            all_train_omics = np.vstack(
                [
                    split_data[k]["train"][pat]["x_omic"]
                    for pat in split_data[k]["train"]
                ]
            )
            scaler = StandardScaler().fit(all_train_omics)
            for split in ["train", "test"]:
                for pat in split_data[k][split]:
                    split_data[k][split][pat]["x_omic"] = scaler.transform(
                        split_data[k][split][pat]["x_omic"].reshape(1, -1)
                    ).flatten()

    return split_data


def get_all_dataset(
    data_dir: str = "./data",
    use_rnaseq: bool = False,
    rm_missing_omics: bool = True,
    rm_missing_grade: bool = True,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Loads the raw patient data/labels and aligns with omics.
    Options for removing/imputing missing data depending on task.

    Args:
        data_dir (str): Directory containing raw data
        use_rnaseq (bool): Whether to include RNA data
        rm_missing_omics (bool): Whether to remove patients with missing omics
        rm_missing_grade (bool): Whether to remove patients with missing grade
        verbose (bool): Print processing steps
    """
    labels = [
        "Grade",
        "censored",
        "Survival months",
        "Event",
    ]

    all_dataset = pd.read_csv(f"{data_dir}/omics/all_dataset.csv").drop(
        "indexes", axis=1
    )
    all_dataset.set_index("TCGA ID", inplace=True)

    all_grade = pd.read_csv(f"{data_dir}/omics/grade_data.csv")

    all_grade.set_index("TCGA ID", inplace=True)

    all_grade["Vital status"] = all_grade["Vital status"].map(
        {"Alive": 1, "Deceased": 0}
    )

    all_dataset = all_dataset.sort_index()
    all_grade = all_grade.sort_index()
    # Sanity checks
    assert pd.Series(all_dataset.index).equals(pd.Series(all_grade.index))
    assert pd.Series(all_dataset["censored"]).equals(all_grade["Vital status"])
    assert pd.Series(all_dataset["Survival months"]).equals(
        all_grade["Time to last followup or death (months)"]
    )

    # assert pd.Series(all_dataset.index).equals(pd.Series(sorted(all_grade.index)))
    all_dataset = all_dataset.join(
        all_grade[["Histology", "Grade", "Molecular subtype"]],
        how="inner",
        on="TCGA ID",
    )

    all_dataset = all_dataset.drop(["Histology", "Molecular subtype"], axis=1)
    all_dataset["Grade"] = all_dataset["Grade"] - 2
    all_dataset["Event"] = 1 - all_dataset["censored"]

    if use_rnaseq:
        print("Adding RNAseq data")
        gbm = pd.read_csv(
            f"{data_dir}/omics/mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt",
            sep="\t",
            skiprows=1,
            index_col=0,
        )
        lgg = pd.read_csv(
            f"{data_dir}/omics/mRNA_Expression_Zscores_RSEM.txt",
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
        print(
            f"Removing {all_dataset.shape[0] - glioma_RNAseq.shape[0]} patients with missing RNAseq data"
        ) if verbose else None
        all_dataset = all_dataset.join(glioma_RNAseq, how="inner")

    # Impute or remove missing data
    if rm_missing_grade:
        print(
            f"Removing {all_dataset['Grade'].isna().sum()} patients with missing grade"
        ) if verbose else None
        all_dataset = all_dataset[all_dataset["Grade"].notna()]
    else:
        # Impute value is irrelevant as grade is a label not a feature
        print(f"Imputing {all_dataset['Grade'].isna().sum()} missing grades with 1") if verbose else None
        all_dataset["Grade"] = all_dataset["Grade"].fillna(1)

    if rm_missing_omics:
        print(
            f"Removing {all_dataset.isna().any(axis=1).sum()} patients with missing omics"
        ) if verbose else None
        # NOTE: This is handled differently to the paper. There are 3 patients with no omics data,
        # but as they have a moltype they are imputed for omic models (in grade classification tasks).
        # Median imputation for omics means all 1s. As such they are removed here.
        all_dataset = all_dataset[all_dataset.notna().all(axis=1)]
    else:
        print(
            f"Imputing missing omics with median in {all_dataset.isna().any(axis=1).sum()} patients"
        ) if verbose else None
        for col in all_dataset.drop(labels, axis=1).columns:
            all_dataset[col] = all_dataset[col].fillna(all_dataset[col].median())

    print(f"Total patients: {all_dataset.shape[0]}") if verbose else None

    return labels, all_dataset
