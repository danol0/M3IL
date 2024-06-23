from argparse import Namespace
from typing import Dict

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Batch
from torchvision import transforms


def define_dataset(opt) -> Dataset:
    if opt.mil == "PFS":
        return psuedoSupervised_TCGADataset
    elif opt.mil in ("global", "local"):
        return MMMIL_TCGADataset
    else:
        raise NotImplementedError(f"MIL type {opt.mil} not implemented.")


def get_transforms() -> transforms.Compose:
    tr = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomCrop(512),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return tr


class MMMILDataset(Dataset):
    def __init__(self, data: Dict) -> None:
        """
        Generic multimodal multiple instance learning dataset.
        Expects a nested data dictionary of the form:
        bag_i: {mode_1: tensor, mode_2: tensor, ..., mode_n: tensor, y: bag_label}
        """
        self.data = data
        self.bagnames = list(self.data.keys())

    def __getitem__(self, idx: int) -> tuple:
        bag = self.bagnames[idx]
        return tuple(self.data[bag].values())


# --- Dataset Class ---
class MMMIL_TCGADataset(Dataset):
    def __init__(self, data: Dict, opt: Namespace) -> None:
        """
        MIL dataset for path, graph and omic TCGA data bagged at patient level.
        Returns all data for a given patient on each call.
        Expects a nested data dictionary of the form:
        patient_id {
            x_omic: omic data tensor
            x_path: pre-extracted histology feature tensor or list of image paths
            x_graph: list of graph data paths
            y: list of event, time, grade labels
        }

        Args:
            data (Dict): Patient data dictionary
            opt (Namespace): Command line arguments
        """
        self.model = opt.model
        self.data = data
        self.patnames = list(self.data.keys())
        self.pre_encoded_path = opt.pre_encoded_path
        self.transform = get_transforms()

    def __getitem__(self, idx: int) -> tuple:
        pname = self.patnames[idx]
        # Only loads necessary data to improve performance
        omic, path, graph = 0, 0, 0

        if "omic" in self.model:
            omic = torch.tensor(self.data[pname]["x_omic"], dtype=torch.float32)

        if "path" in self.model:
            if self.pre_encoded_path:
                path = self.data[pname]["x_path"]
                path = torch.tensor(path, dtype=torch.float32).squeeze(1)
            else:
                raise NotImplementedError(
                    "Path network not implemented at patient level"
                )
                path = [Image.open(p) for p in self.data[pname]["x_path"]]
                path = torch.stack([self.transform(p.convert("RGB")) for p in path])

        if "graph" in self.model:
            graph = [torch.load(g) for g in self.data[pname]["x_graph"]]
            graph = Batch.from_data_list(graph)

        event, time, grade = [
            torch.tensor(self.data[pname]["y"][i], dtype=torch.long) for i in range(3)
        ]
        return (omic, path, graph, event, time, grade, pname)

    def __len__(self):
        return len(self.patnames)


class psuedoSupervised_TCGADataset(Dataset):
    def __init__(self, data: Dict, opt: Namespace) -> None:
        """
        Defines a multimodal pathology dataset of instance level combinations.
        Returns a single instance from each modality on each call, paired with the bag label.
        On init, a list of instance combinations that index the data dictionary is created.
        This is based on the assumption that there are 9 path instances for each graph instance,
        and that the x_path and x_graph lists are ordered accordingly.
        Patches are only paired with their parent graph as done in the paper.
        """
        self.model = opt.model
        self.data = data
        self.pre_encoded_path = opt.pre_encoded_path
        self.transform = get_transforms()

        combinations = []
        for pat in self.data.keys():
            num_path = len(self.data[pat]["x_path"]) if "path" in self.model else 1
            num_graph = len(self.data[pat]["x_graph"]) if "graph" in self.model else 1
            if "pathgraph" in self.model:
                assert num_path == num_graph * 9
                combinations.extend([(pat, i, i // 9) for i in range(num_path)])
            else:
                combinations.extend(
                    [(pat, i, j) for i in range(num_path) for j in range(num_graph)]
                )
        self.combinations = combinations

    def __getitem__(self, idx):
        pat, path_idx, graph_idx = self.combinations[idx]
        omic, path, graph = 0, 0, 0

        if "omic" in self.model:
            omic = torch.tensor(self.data[pat]["x_omic"], dtype=torch.float32)

        if "path" in self.model:
            if self.pre_encoded_path:
                path = torch.tensor(
                    self.data[pat]["x_path"][path_idx], dtype=torch.float32
                ).squeeze(0)
            else:
                path = self.transform(
                    Image.open(self.data[pat]["x_path"][path_idx]).convert("RGB")
                )

        if "graph" in self.model:
            graph = [torch.load(self.data[pat]["x_graph"][graph_idx])]
            graph = Batch.from_data_list(graph)

        event, time, grade = [
            torch.tensor(self.data[pat]["y"][k], dtype=torch.long) for k in range(3)
        ]
        return (omic, path, graph, event, time, grade, pat)

    def __len__(self):
        return len(self.combinations)


# --- Collate functions ---
def define_collate_fn(opt: Namespace) -> callable:
    # Only path requires collating (graphs are collated through nested batches)
    if opt.mil == "PFS" or "path" not in opt.model:
        return lambda batch: mixed_collate(batch, opt.device)
    else:
        if opt.collate == "min":
            return lambda batch: select_min(batch, opt.device)
        elif opt.collate == "pad":
            return lambda batch: zero_pad(batch, opt.device)
        else:
            raise NotImplementedError(f"Collate function {opt.collate} not implemented")


def mixed_collate(batch: list, device: torch.device) -> tuple:
    """Applies graph-specific collate operations."""

    collated_batch = (
        (
            prepare_graph_batch(samples, device)
            if isinstance(samples[0], Batch)
            else default_collate(samples)
        )
        for samples in zip(*batch)
    )

    return tuple(
        data.to(device) if isinstance(data, torch.Tensor) else data
        for data in collated_batch
    )


def prepare_graph_batch(graph: Batch, device: torch.device) -> tuple:
    """Batches graphs, and creates a tensor of patient indices (for patient-level MIL)."""

    pat_idxs = torch.repeat_interleave(
        torch.arange(len(graph)), torch.tensor([len(g.ptr) - 1 for g in graph])
    )
    graph = Batch.from_data_list(graph)
    return graph.to(device), pat_idxs.to(device)


# --- Handling variable image input sequence lengths ---
def zero_pad(batch: list, device: torch.device) -> tuple:
    """Pads image sequences to the maximum length in the batch."""

    omic, path, graph, event, time, grade, pname = zip(*batch)
    path = pad_sequence(path, batch_first=True, padding_value=0)
    batch = list(zip(omic, path, graph, event, time, grade, pname))
    return mixed_collate(batch, device)


def select_min(batch: list, device: torch.device) -> tuple:
    """Selects a random subset of images from each patient in the batch."""

    omic, path, graph, event, time, grade, pname = zip(*batch)
    if len(path) > 1:
        min_imgs = min([img.size(0) for img in path])
        path = [img[torch.randperm(img.size(0))[:min_imgs]] for img in path]
    batch = list(zip(omic, path, graph, event, time, grade, pname))
    return mixed_collate(batch, device)
