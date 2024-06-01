import torch
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Batch
from torchvision import transforms


def define_dataset(opt):
    if opt.mil in ("instance", "paper"):
        return instanceLevelDataset
    elif opt.mil == "pat":
        return patientLevelDataset
    else:
        raise NotImplementedError(f"MIL type {opt.mil} not implemented")


def get_transforms():
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


# --- Dataset Class ---
class instanceLevelDataset(Dataset):
    """Defines a multimodal pathology dataset of instance level combinations."""

    def __init__(self, data, split, opt):
        self.model = opt.model
        self.data = data[split]
        self.vgg = opt.use_vgg
        self.transform = get_transforms()

        combinations = []
        for pat in self.data.keys():
            num_path = len(self.data[pat]["x_path"]) if "path" in self.model else 1
            num_graph = len(self.data[pat]["x_graph"]) if "graph" in self.model else 1
            if opt.mil == "paper" and "pathgraph" in self.model:
                assert num_path == num_graph * 9
                combinations.extend([(pat, i, i // 9) for i in range(num_path)])
            else:
                combinations.extend(
                    [(pat, i, j) for i in range(num_path) for j in range(num_graph)]
                )
        self.combinations = combinations

    def __getitem__(self, idx):
        pat, i, j = self.combinations[idx]

        omic, path, graph = 0, 0, 0

        if "omic" in self.model:
            omic = torch.tensor(self.data[pat]["x_omic"], dtype=torch.float32)

        if "path" in self.model:
            if self.vgg:
                path = torch.tensor(
                    self.data[pat]["x_path"][i], dtype=torch.float32
                ).squeeze(0)
            else:
                path = self.transform(
                    Image.open(self.data[pat]["x_path"][i]).convert("RGB")
                )

        if "graph" in self.model:
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
        self.model = opt.model
        self.data = data[split]
        self.patnames = list(self.data.keys())
        self.vgg = opt.use_vgg
        self.transform = get_transforms()

    def __getitem__(self, idx):
        pname = self.patnames[idx]

        omic, path, graph = 0, 0, 0

        if "omic" in self.model:
            omic = torch.tensor(self.data[pname]["x_omic"], dtype=torch.float32)

        if "path" in self.model:
            if self.vgg:
                path = self.data[pname]["x_path"]
                path = torch.tensor(path, dtype=torch.float32).squeeze(1)
            else:
                raise NotImplementedError("Path not implemented at patient level")
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


# --- Collate functions ---
def define_collate_fn(opt):
    # Only path requires collating
    if opt.mil in ("instance", "paper") or "path" not in opt.model:
        return lambda batch: mixed_collate(batch, opt.device)
    else:
        if opt.collate == "min":
            return lambda batch: select_min(batch, opt.device)
        elif opt.collate == "pad":
            return lambda batch: pad2max(batch, opt.device)
        else:
            raise NotImplementedError(f"Collate function {opt.collate} not implemented")


def mixed_collate(batch, device):
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


def prepare_graph_batch(graph, device):
    pat_idxs = torch.repeat_interleave(
        torch.arange(len(graph)), torch.tensor([len(g.ptr) - 1 for g in graph])
    )
    graph = Batch.from_data_list(graph)
    return graph.to(device), pat_idxs.to(device)


# --- Handling variable image input sequence lengths ---
def select_min(batch, device):
    omic, path, graph, event, time, grade, pname = zip(*batch)

    # find minimum number of images
    if len(path) > 1:
        min_imgs = min([img.size(0) for img in path])
        # select min_imgs random images from each patient
        path = [img[torch.randperm(img.size(0))[:min_imgs]] for img in path]

    batch = list(zip(omic, path, graph, event, time, grade, pname))
    return mixed_collate(batch, device)


def pad2max(batch, device):
    omic, path, graph, event, time, grade, pname = zip(*batch)

    # Find max number of images
    max_imgs = max([img.size(0) for img in path])
    # 0 pad
    path = [
        torch.cat([img, img.new_zeros(max_imgs - img.size(0), *img.size()[1:])])
        for img in path
    ]
    batch = list(zip(omic, path, graph, event, time, grade, pname))
    return mixed_collate(batch, device)
