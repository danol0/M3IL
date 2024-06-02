import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_splits(opt):
    """Gets the data splits for the PNAS dataset."""
    data_dir = opt.data_dir
    labels, dataset = get_all_dataset(
        use_rnaseq=opt.rna,
        rm_missing_omics=1 if "omic" in opt.model else 0,
        rm_missing_grade=1 if opt.task in ["multi", "grad"] else 0,
        data_dir=data_dir,
    )
    split_data = {}

    pnas_splits = pd.read_csv(f"{data_dir}/splits/pnas_splits.csv")
    pnas_splits.columns = ["TCGA ID"] + [k for k in range(1, 16)]
    pnas_splits.set_index("TCGA ID", inplace=True)
    pnas_splits = pnas_splits.applymap(lambda x: x.lower())

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
                vgg_feats = pickle.load(open(f"{data_dir}/path/resnet_features.pkl", "rb"))
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
                                    vgg_feats[str(k)][patch] if opt.use_vggnet else vgg_feats[patch]
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
                for pat in pnas_splits.index[
                    (pnas_splits[k] == split) & (pnas_splits.index.isin(dataset.index))
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
    data_dir="./data",
    use_rnaseq=False,
    rm_missing_omics=True,
    rm_missing_grade=True,
):
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

    assert pd.Series(all_dataset.index).equals(pd.Series(sorted(all_grade.index)))
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
        all_dataset = all_dataset.join(glioma_RNAseq, how="inner")

    # Impute or remove missing data
    if rm_missing_grade:
        print(
            f"Removing {all_dataset['Grade'].isna().sum()} patients with missing grade"
        )
        all_dataset = all_dataset[all_dataset["Grade"].notna()]
    else:
        print(f"Imputing {all_dataset['Grade'].isna().sum()} missing grades with 1")
        all_dataset["Grade"] = all_dataset["Grade"].fillna(1)

    if rm_missing_omics:
        print(
            f"Removing {all_dataset.isna().any(axis=1).sum()} patients with missing omics"
        )
        # NOTE: This is handled differently to the paper. 3 patients with missing omics but
        # not missing moltype. They are imputed in the paper, but removed here.
        # TODO: Check this and write it up
        all_dataset = all_dataset[all_dataset.notna().all(axis=1)]
    else:
        print(
            f"Imputing missing omics with median in {all_dataset.isna().any(axis=1).sum()} patients"
        )
        for col in all_dataset.drop(labels, axis=1).columns:
            all_dataset[col] = all_dataset[col].fillna(all_dataset[col].median())

    print(f"Saving cleaned dataset to {data_dir}/omics/cleaned_dataset.csv")
    all_dataset.to_csv(f"{data_dir}/omics/cleaned_dataset.csv")
    print(f"Total patients: {all_dataset.shape[0]}")

    return labels, all_dataset
