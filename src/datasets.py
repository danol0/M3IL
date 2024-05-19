import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
from torch_geometric.data import Batch
from torch.utils.data.dataloader import default_collate


def define_dataset(opt):
    if opt.mil in ("instance", "paper"):
        return instanceLevelDataset
    elif opt.mil == "pat":
        return patientLevelDataset
    else:
        raise NotImplementedError(f"MIL type {opt.mil} not implemented")


# --- Dataset Class ---
# TODO: Implement paperDataset


class instanceLevelDataset(Dataset):
    """Defines a multimodal pathology dataset of instance level combinations."""

    def __init__(self, data, split, opt):
        self.mode = opt.model
        self.data = data[split]

        combinations = []
        for pat in self.data.keys():
            num_path = self.data[pat]["x_path"].shape[0] if "path" in self.mode else 1
            num_graph = len(self.data[pat]["x_graph"]) if "graph" in self.mode else 1
            if opt.mil == "paper" and "pathgraph" in self.mode:
                assert num_path == num_graph * 9
                combinations.extend([(pat, i, i // 9) for i in range(num_path)])
            combinations.extend(
                [(pat, i, j) for i in range(num_path) for j in range(num_graph)]
            )
        self.combinations = combinations

    def __getitem__(self, idx):
        pat, i, j = self.combinations[idx]

        omic, path, graph = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        if "omic" in self.mode:
            omic = torch.tensor(self.data[pat]["x_omic"], dtype=torch.float32)

        if "path" in self.mode:
            path = torch.tensor(
                self.data[pat]["x_path"][i], dtype=torch.float32
            ).squeeze(0)

        if "graph" in self.mode:
            graph = [torch.load(self.data[pat]["x_graph"][j])]
            graph = Batch.from_data_list(graph)

        event, time, grade = [
            torch.tensor(self.data[pat]["y"][k], dtype=torch.long) for k in range(3)
        ]

        return (omic, path, graph, event, time, grade, pat)

    def __len__(self):
        return len(self.combinations)


class patientLevelDataset(Dataset):
    """Defines a multimodal pathology dataset bagged at the patient level."""

    def __init__(self, data, split, opt):
        self.mode = opt.model
        self.data = data[split]
        self.patnames = list(self.data.keys())

    def __getitem__(self, idx):
        pname = self.patnames[idx]
        # Empty tensors are compatible with downstream operations
        omic, path, graph = torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        if "omic" in self.mode:
            omic = torch.tensor(self.data[pname]["x_omic"], dtype=torch.float32)

        if "path" in self.mode:
            path = self.data[pname]["x_path"]
            path = torch.tensor(path, dtype=torch.float32).squeeze(1)

        if "graph" in self.mode:
            graph = [torch.load(g) for g in self.data[pname]["x_graph"]]
            graph = Batch.from_data_list(graph)

        event, time, grade = [
            torch.tensor(self.data[pname]["y"][i], dtype=torch.long) for i in range(3)
        ]
        return (omic, path, graph, event, time, grade, pname)

    def __len__(self):
        return len(self.patnames)


# --- Collate functions ---
def define_collate_fn(opt):
    if opt.mil in ("instance", "paper"):
        return default_collate
    else:
        if opt.collate == "min":
            return select_min
        elif opt.collate == "pad":
            return pad2max
        else:
            raise NotImplementedError(f"Collate function {opt.collate} not implemented")


def prepare_graph_batch(graph):
    n_graphs = torch.Tensor([0])
    if not isinstance(graph[0], torch.Tensor):
        n_graphs = torch.Tensor([len(g.ptr) - 1 for g in graph]).long()
        graph = Batch.from_data_list(graph)
    return graph, n_graphs


def select_min(batch):
    omic, path, graph, event, time, grade, pname = zip(*batch)

    # find minimum number of images
    if len(path) > 1:
        min_imgs = min([img.size(0) for img in path])
        # select min_imgs random images from each patient
        path = [img[torch.randperm(img.size(0))[:min_imgs]] for img in path]

    return (
        torch.stack(omic),
        torch.stack(path),
        prepare_graph_batch(graph),
        torch.stack(event),
        torch.stack(time),
        torch.stack(grade),
        pname,
    )


def pad2max(batch):
    omic, path, graph, event, time, grade, pname = zip(*batch)

    # find max number of images
    max_imgs = max([img.size(0) for img in path])
    # 0 pad
    path = [
        torch.cat([img, img.new_zeros(max_imgs - img.size(0), *img.size()[1:])])
        for img in path
    ]

    return (
        torch.stack(omic),
        torch.stack(path),
        prepare_graph_batch(graph),
        torch.stack(event),
        torch.stack(time),
        torch.stack(grade),
        pname,
    )


# --- Data loading ---
def get_splits(opt, data_dir="./data"):
    rmomics = 1 if "omic" in opt.model else 0
    rmgrade = 1 if opt.task in ["multi", "grad"] else 0
    labels, dataset = getCleanAllDataset(
        use_rnaseq=opt.rna, rm_missing_omics=rmomics, rm_missing_grade=rmgrade
    )
    split_data = {}

    pnas_splits = pd.read_csv(f"{data_dir}/splits/pnas_splits.csv")
    pnas_splits.columns = ["TCGA ID"] + [k for k in range(1, 16)]
    pnas_splits.set_index("TCGA ID", inplace=True)
    pnas_splits = pnas_splits.map(lambda x: x.lower())

    vgg_ftype = "surv" if opt.task == "multi" else opt.task
    vgg_feats = pickle.load(open(f"{data_dir}/vgg_features_{vgg_ftype}.pkl", "rb"))

    pat2roi = defaultdict(list)
    roi2patch = defaultdict(list)
    for roi_fname in os.listdir(f"{data_dir}/graphs/"):
        pat2roi[roi_fname[:12]].append(roi_fname.rstrip(".pt"))
        # We do this with nested loops to preserve the order of the patches
        # This allows us to match the patches to parent graphs as done in the paper
        for img_fname in vgg_feats.keys():
            if img_fname.startswith(roi_fname.rstrip(".pt")):
                roi2patch[roi_fname.rstrip(".pt")].append(img_fname)

    for k in range(1, opt.folds + 1):
        split_data[k] = {
            split: {
                pat: {
                    "x_omic": dataset.loc[pat].drop(labels).values.astype(np.float32),
                    "x_path": np.stack(
                        [
                            vgg_feats[patch]
                            for roi in pat2roi[pat]
                            for patch in roi2patch[roi]
                        ]
                    ),
                    "x_graph": [f"{data_dir}/graphs/{roi}.pt" for roi in pat2roi[pat]],
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


# TODO: allow for different moltype ignore etc
def getCleanAllDataset(
    data_dir="./data/omics",
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

    all_dataset = pd.read_csv(f"{data_dir}/all_dataset.csv").drop("indexes", axis=1)
    all_dataset.set_index("TCGA ID", inplace=True)

    all_grade = pd.read_csv(f"{data_dir}/grade_data.csv")

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
        all_dataset = all_dataset[all_dataset.notna().all(axis=1)]
    else:
        print(
            f"Imputing missing omics with median in {all_dataset.isna().any(axis=1).sum()} patients"
        )
        for col in all_dataset.drop(labels, axis=1).columns:
            all_dataset[col] = all_dataset[col].fillna(all_dataset[col].median())

    print(f"Saving cleaned dataset to {data_dir}/cleaned_dataset.csv")
    all_dataset.to_csv(f"{data_dir}/cleaned_dataset.csv")
    print(f"Total patients: {all_dataset.shape[0]}")

    return labels, all_dataset
