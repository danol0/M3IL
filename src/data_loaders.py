from PIL import Image
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms
import numpy as np


class omicDataset(Dataset):
    def __init__(self, data, split):
        self.data = data[split]
        self.patnames = list(self.data.keys())

    def __getitem__(self, idx):
        pname = self.patnames[idx]
        omics = torch.tensor(self.data[pname]["x_omic"]).type(torch.FloatTensor)

        imgs = self.data[pname]["x_path"]
        idxs = np.random.choice(len(imgs), 9, replace=False)
        imgs = torch.tensor(imgs[idxs]).type(torch.FloatTensor).squeeze(1)

        censored, time, grade = [
            torch.tensor(self.data[pname]["y"][i]).type(torch.LongTensor)
            for i in range(3)
        ]
        return (omics, imgs, censored, time, grade)

    def __len__(self):
        return len(self.patnames)


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
# class PathgraphomicDataset(Dataset):
#     def __init__(self, opt, data, split):
#         """
#         Args:
#             X = data
#             e = overall survival event
#             t = overall survival in months
#         """
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
#             "path": (lambda: (self.transforms(Image.open(self.X_path[index]).convert('RGB')), 0, 0)),
#             "graph": (lambda: (0, torch.load(self.X_grph[index]), 0)),
#             "omic": (lambda: (0, 0, torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#         } if not self.vgg else {
#             "path": (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), 0, 0)),
#             "graph": (lambda: (0, torch.load(self.X_grph[index]), 0)),
#             "omic": (lambda: (0, 0, torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#             "pathomic": (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), 0, torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#             "graphomic": (lambda: (0, torch.load(self.X_grph[index]), torch.tensor(self.X_omic[index]).type(torch.FloatTensor))),
#             "pathgraph": (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), torch.load(self.X_grph[index]), 0)),
#             "pathgraphomic": (lambda: (torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0), torch.load(self.X_grph[index]), torch.tensor(self.X_omic[index]).type(torch.FloatTensor)))
#         }
#         return (*loaders.get(self.mode, (lambda: (None, None, None)))(), single_e, single_t, single_g)

#     def __len__(self):
#         return len(self.X_path)
