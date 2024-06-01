import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch_geometric.nn import (
    MeanAggregation,
    SAGEConv,
    SAGPooling,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import scatter, softmax


def dfs_freeze(model, freeze):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = not freeze
        dfs_freeze(child, freeze)


class BaseEncoder(nn.Module):
    def __init__(self, fdim):
        """Base encoder for downstream tasks acting on a processed feature vector."""
        super().__init__()
        self.grade_clf = nn.Sequential(nn.Linear(fdim, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(fdim, 1), nn.Sigmoid())
        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

    def forward(self, x):
        grade = self.grade_clf(x)
        hazard = self.hazard_clf(x) * self.output_range + self.output_shift
        return x, grade, hazard

    def freeze(self, freeze):
        dfs_freeze(self, freeze)


# --- Pooling ---
class MaskedAttentionPool(nn.Module):
    """Masked attention pooling over dim 1 for 0 padded inputs.
    Source: https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py"""

    def __init__(self, fdim=32, hdim=16, dropout=0.25):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(fdim, hdim), nn.Sigmoid(), nn.Dropout(dropout))
        self.b = nn.Sequential(nn.Linear(fdim, hdim), nn.Tanh(), nn.Dropout(dropout))
        self.c = nn.Linear(hdim, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        A = a.mul(b)
        A = self.c(A)
        pad = torch.all(x == 0, dim=-1)
        A[pad] = -float("Inf")
        A = F.softmax(A, dim=1)
        return torch.sum(A * x, dim=1)


class MaskedMeanPool(nn.Module):
    """Masked mean pooling over dim 1 for 0 padded inputs."""

    def forward(self, x):
        mask = torch.any(x != 0, dim=-1)
        assert torch.all(mask.any(dim=1)), "Empty input detected."
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)


class GraphAttentionPool(nn.Module):
    """Attention pooling for graph batches.
    Source: https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/nn/glob/attention.html
    """

    def __init__(self, fdim=32, hdim=16, dropout=0.25):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(fdim, hdim), nn.Sigmoid(), nn.Dropout(dropout))
        self.b = nn.Sequential(nn.Linear(fdim, hdim), nn.Tanh(), nn.Dropout(dropout))
        self.c = nn.Linear(hdim, 1)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        a = self.a(x)
        b = self.b(x)
        A = a.mul(b)
        A = self.c(A).view(-1, 1)
        A = softmax(A, batch, size)
        assert A.dim() == x.dim() and A.size(0) == x.size(0)
        out = scatter(src=(A * x), index=batch, reduce="add", dim_size=size)
        return out


# --- Omic ---
class FFN(BaseEncoder):
    def __init__(self, opt, xdim=80, fdim=32, dropout_layer=0):
        super().__init__(fdim)
        hidden = [64, 48, 32, fdim]

        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Linear(xdim if i == 0 else hidden[i - 1], hidden[i]))
            layers.append(nn.ELU())
            if i >= dropout_layer:
                layers.append(nn.AlphaDropout(p=opt.dropout, inplace=False))
            else:
                layers.append(nn.AlphaDropout(p=0, inplace=False))

        self.encoder = nn.Sequential(*layers)

    def forward(self, **kwargs):
        x = kwargs["x_omic"]
        x = self.encoder(x)
        return super().forward(x)


# --- Path ---
class VGGNet(BaseEncoder):
    def __init__(self, vgg_layers, opt, fdim=32):
        super().__init__(fdim)
        self.conv_layers = vgg_layers
        # Using avgpool for MPS compatibility
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, fdim),
            nn.ReLU(True),
            nn.Dropout(0.05),
        )

        if opt.mil == "pat":
            raise NotImplementedError("Bagging not implemented for path model.")

        dfs_freeze(self.conv_layers, freeze=True)

    def forward(self, **kwargs):
        x = kwargs["x_path"]
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return super().forward(x)


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def get_vgg19(opt, fdim=32):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M"]
    cfg += [512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    model = VGGNet(make_layers(cfg), opt, fdim=fdim)

    pretrained_dict = load_state_dict_from_url(
        "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth", progress=True
    )

    for key in list(pretrained_dict.keys()):
        if "classifier" in key:
            pretrained_dict.pop(key)
        if key.startswith("features."):
            pretrained_dict[key[9:]] = pretrained_dict.pop(key)

    print("Initializing VGG19 Weights")
    model.conv_layers.load_state_dict(pretrained_dict, strict=True)

    return model


# --- Graph ---
class GNN(BaseEncoder):
    def __init__(self, opt, xdim=1036, hdim=128, fdim=32):
        super().__init__(fdim)

        hidden = [xdim, hdim, hdim]

        self.convs = torch.nn.ModuleList(
            [SAGEConv(in_channels, hdim) for in_channels in hidden]
        )
        self.pools = torch.nn.ModuleList(
            [SAGPooling(hdim, ratio=0.2) for _ in range(len(hidden))]
        )
        self.encoder = nn.Sequential(
            torch.nn.Linear(hdim * 2, hdim),
            nn.ReLU(True),
            nn.Dropout(p=opt.dropout),
            torch.nn.Linear(hdim, fdim),
            nn.ReLU(True),
        )
        self.aggregate = None
        if opt.mil == "pat":
            if "qbt" in opt.model:
                self.aggregate = self.collate_graphs
            else:
                self.aggregate = (
                    GraphAttentionPool(xdim=fdim, hdim=16, dropout=opt.dropout)
                    if opt.attn_pool
                    else MeanAggregation()
                )

    def normalize_graphs(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        data.x = data.x.type(torch.FloatTensor).to(data.x.device)
        data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(
            data.edge_attr.device
        )
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def collate_graphs(self, x, batch):
        """Converts graph embeddings of shape [n_graphs, fdim] to tensor of
        shape [batch_size, max(graphs_per_pat), fdim], with 0 padding. For use with QBT.
        """
        bincount = batch.bincount()
        cumsum = bincount.cumsum(0)
        batch_size = batch[-1].item() + 1
        x = x.squeeze(1)
        patch_graphs = torch.zeros(
            batch_size,
            bincount.max(),
            x.size(-1),
            device=x.device,
        )
        for i, (start, end) in enumerate(zip(cumsum - bincount, cumsum)):
            patch_graphs[i, : end - start] = x[start:end]
        return patch_graphs

    def forward(self, **kwargs):
        data, pat_idxs = kwargs["x_graph"]
        data = self.normalize_graphs(data)
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        xs = []
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index))
            x, edge_index, edge_attr, batch, _, _ = pool(
                x, edge_index, edge_attr, batch
            )
            xs.append(
                torch.cat(
                    [global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1
                )
            )

        x = torch.sum(torch.stack(xs), dim=0)
        x = self.encoder(x)
        if self.aggregate:
            x = self.aggregate(x, pat_idxs)
        return super().forward(x)
