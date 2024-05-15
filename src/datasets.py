import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


# --- Dataset Class ---
class pathologyDataset(Dataset):
    def __init__(self, data, split):
        self.data = data[split]
        self.patnames = list(self.data.keys())

    def __getitem__(self, idx):
        pname = self.patnames[idx]
        omics = torch.tensor(self.data[pname]["x_omic"]).type(torch.FloatTensor)

        imgs = self.data[pname]["x_path"]
        imgs = torch.tensor(imgs).type(torch.FloatTensor).squeeze(1)

        event, time, grade = [
            torch.tensor(self.data[pname]["y"][i]).type(torch.LongTensor)
            for i in range(3)
        ]
        return (omics, imgs, event, time, grade, pname)

    def __len__(self):
        return len(self.patnames)


# --- Collate functions ---
def select_min(batch):
    omics, imgs, event, time, grade, pname = zip(*batch)

    # find minimum number of images
    min_imgs = min([img.size(0) for img in imgs])
    # select min_imgs random images
    idxs = np.random.choice(min_imgs, 9, replace=False)
    imgs = [img[idxs] for img in imgs]
    return (
        torch.stack(omics),
        torch.stack(imgs),
        torch.stack(event),
        torch.stack(time),
        torch.stack(grade),
        pname,
    )


def pad2max(batch):
    omics, imgs, event, time, grade = zip(*batch)

    # find max number of images
    max_imgs = max([img.size(0) for img in imgs])
    # 0 pad
    imgs = [
        torch.cat([img, img.new_zeros(max_imgs - img.size(0), *img.size()[1:])])
        for img in imgs
    ]
    return (
        torch.stack(omics),
        torch.stack(imgs),
        torch.stack(event),
        torch.stack(time),
        torch.stack(grade),
    )


# --- Data loading ---
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
def getCleanAllDataset(data_dir="./data/omics", use_rnaseq=False):
    metadata = [
        "Histology",
        "Grade",
        "Molecular subtype",
        "censored",
        "Survival months",
        "Event",
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


# class bagLoader(Dataset):
#     def __init__(self, opt, data, split):
#         self.data = data[split]
#         self.patnames = list(self.data.keys())

#     def __getitem__(self, idx):
#         pname = self.patnames[idx]
#         # select 9 random images
#         imgs = self.data[pname]['images']
#         idxs = np.random.choice(len(imgs), 9, replace=False)
#         imgs = torch.tensor(imgs[idxs]).type(torch.FloatTensor).squeeze(1)

#         omics = torch.tensor(self.data[pname]['omics']).type(torch.FloatTensor)

#         e = torch.tensor(self.data[pname]['labels'][0]).type(torch.FloatTensor)
#         t = torch.tensor(self.data[pname]['labels'][1]).type(torch.FloatTensor)
#         g = torch.tensor(self.data[pname]['labels'][2]).type(torch.LongTensor)

#         return (imgs, 0, omics, e, t, g)  # 9, null, 1, 1, 1, 1

#     def __len__(self):
#         return len(self.patnames)


# ################
# # Dataset Class
# ################

# # data is a pkl file made in make_splits with the following keys:
# #    - data_pd : pandas dataframe with all the data
# #    - pat2img : dictionary with keys as TCGA IDs and values as a list of ROIs associated with that patient
# #    - img_fnames : list of all ROIs
# #    - cv_splits : nested dicts:
# #        1. keys are the split number (0-15)
# #        2. values are dicts with keys 'train', 'test'
# #        3. values of 'train' and 'test' are dicts with keys 'x_patname', 'x_path', 'x_grph', 'x_omic', 'e', 't', 'g'
# class PathgraphpathologyDataset(Dataset):
#     def __init__(self, opt, data, split):
#         '''
#         Args:
#             X = data
#             e = overall survival event
#             t = overall survival in months
#         '''
#         self.X_path = data[split]['x_path']
#         self.X_grph = data[split]['x_grph']
#         self.X_omic = data[split]['x_omic']
#         self.e = data[split]['e']
#         self.t = data[split]['t']
#         self.g = data[split]['g']
#         self.mode = opt.mode
#         self.vgg = opt.use_vgg_features
#         self.patch = 1 if opt.use_vgg_features else 0

#         self.transforms = (
#             transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                 ]) if self.patch else transforms.Compose([
#                     transforms.RandomHorizontalFlip(0.5),
#                     transforms.RandomVerticalFlip(0.5),
#                     transforms.RandomCrop(512),
#                     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                 ])
#         )

#     def __getitem__(self, index):
#         single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
#         single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
#         single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

#         loaders = {
#             'path': (lambda: (self.transforms(Image.open(self.X_path[index]).convert('RGB')), 0, 0)),
#             'graph': (lambda: (0, torch.load(self.X_grph[index]), 0)),
#             'omic': (lambda: (0, 0, torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#         } if not self.vgg else {
#             'path': (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), 0, 0)),
#             'graph': (lambda: (0, torch.load(self.X_grph[index]), 0)),
#             'omic': (lambda: (0, 0, torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#             'pathomic': (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), 0, torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#             'graphomic': (lambda: (0, torch.load(self.X_grph[index]), torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#             'pathgraph': (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), torch.load(self.X_grph[index]), 0)),
#             'pathgraphomic': (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), torch.load(self.X_grph[index]), torch.tensor(self.X_omic[index]).type(torch.FloatTensor)))
#         }
#         return (*loaders.get(self.mode, (lambda: (None, None, None)))(), single_e, single_t, single_g)

#     def __len__(self):
#         return len(self.X_path)
