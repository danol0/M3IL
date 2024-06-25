import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.nn import (
    MeanAggregation,
    SAGEConv,
    SAGPooling,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import scatter, softmax


def dfs_freeze(model, freeze: bool) -> None:
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = not freeze
        dfs_freeze(child, freeze)


class BaseEncoder(nn.Module):
    def __init__(self, fdim: int, local: bool = False) -> None:
        """
        Defines output layers for downstream tasks (grade, survival).

        Args:
            fdim (int): Processed feature dimension.
            local (bool): If using local MIL, ie aggregating labels.
        """
        super().__init__()
        self.grade_clf = nn.Sequential(nn.Linear(fdim, 3), nn.LogSoftmax(dim=-1))
        self.hazard_clf = nn.Sequential(nn.Linear(fdim, 1), nn.Sigmoid())
        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))
        self.local = local

    def predict(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        grade = self.grade_clf(x)
        hazard = self.hazard_clf(x) * self.output_range + self.output_shift

        if self.local:
            assert mask is not None, "Mask required for local MIL."
            grade = MaskedMeanPool()(grade, mask)
            hazard = MaskedMeanPool()(hazard, mask)

        return x, grade, hazard

    def freeze(self, freeze: bool) -> None:
        dfs_freeze(self, freeze)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def l1(self) -> torch.Tensor:
        # Placeholder for L1 regularization, overwritten in models where it is enabled
        return torch.tensor(0.0, device=self.output_range.device)


# --- Pooling ---
class BaseAttentionPool(nn.Module):
    def __init__(
        self,
        fdim: int = 32,
        hdim: int = 16,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        """
        Computes a gated attention score over input features.

        Args:
            fdim (int): Input feature dimension.
            hdim (int): Hidden layer dimension.
            dropout (float): Dropout rate.
            temperature (float): Scaling factor.

        Adapted from: https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
        """
        super().__init__()
        self.a = nn.Sequential(nn.Linear(fdim, hdim), nn.Sigmoid(), nn.Dropout(dropout))
        self.b = nn.Sequential(nn.Linear(fdim, hdim), nn.Tanh(), nn.Dropout(dropout))
        self.c = nn.Linear(hdim, 1)
        self.temperature = temperature

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a(x)
        b = self.b(x)
        A = a.mul(b)
        A = self.c(A) / self.temperature
        return A


class MaskedAttentionPool(BaseAttentionPool):
    def __init__(
        self,
        fdim: int = 32,
        hdim: int = 16,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        """
        Performs masked attention pooling over first dimension for 0 padded inputs.
        (batch, samples, features) -> (batch, pooled_features)
        """
        super().__init__(fdim, hdim, dropout, temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.attn(x)
        mask = torch.all(x == 0, dim=-1)
        A[mask] = -float("Inf")
        A = F.softmax(A, dim=1)
        assert A.dim() == x.dim() and A.size(0) == x.size(0)
        return torch.sum(A * x, dim=1)


class GraphAttentionPool(BaseAttentionPool):
    def __init__(
        self,
        fdim: int = 32,
        hdim: int = 16,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        """
        Performs attention pooling of graph features via an index vector.
        (n_graphs, features) -> (batch_size, pooled_features)

        Adapted from: https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/nn/glob/attention.html
        """
        super().__init__(fdim, hdim, dropout, temperature)

    def forward(
        self, x: torch.Tensor, index: torch.Tensor, dim_size: int = None
    ) -> torch.Tensor:
        dim_size = index[-1].item() + 1 if dim_size is None else dim_size
        A = self.attn(x)
        A = softmax(A, index=index, num_nodes=dim_size)
        assert A.dim() == x.dim() and A.size(0) == x.size(0)
        return scatter(src=(A * x), index=index, reduce="add", dim_size=dim_size)


class MaskedMeanPool(nn.Module):
    """Mean pooling over dim 1 for 0 padded inputs."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.any(x != 0, dim=-1)
            assert torch.all(mask.any(dim=1)), "All-zero batch detected."
        else:
            assert x.dim() == mask.dim() + 1 and x.size()[:2] == mask.size()[:2]
            x = x * mask.unsqueeze(-1)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)


# --- Omic ---
class FFN(BaseEncoder):
    def __init__(self, xdim: int = 80, fdim: int = 32, dropout: float = 0.25) -> None:
        """
        Feedforward neural network for tabular omic data.

        Args:
            xdim (int): Input feature dimension.
            fdim (int): Encoded feature dimension.
            dropout (float): Dropout rate.
        """
        super().__init__(fdim)
        hidden = [64, 48, 32, fdim]

        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Linear(xdim if i == 0 else hidden[i - 1], hidden[i]))
            layers.append(nn.ELU())
            layers.append(nn.AlphaDropout(p=dropout, inplace=False))

        self.encoder = nn.Sequential(*layers)

    def get_latents(self, **kwargs: torch.Tensor) -> tuple:
        x = kwargs["x_omic"]
        x = self.encoder(x)
        return x

    def forward(self, **kwargs: torch.Tensor) -> tuple:
        x = self.get_latents(**kwargs)
        return self.predict(x)

    def l1(self) -> torch.Tensor:
        return sum(torch.abs(W).sum() for W in self.parameters())


# --- Graph ---
class GNN(BaseEncoder):
    def __init__(
        self,
        xdim: int = 1036,
        hdim: int = 128,
        fdim: int = 32,
        pool: str = None,
        dropout: int = 0.25,
    ) -> None:
        """
        Graph neural network for encoding histology graphs.

        Args:
            xdim (int): Graph node dimension.
            hdim (int): Hidden layer dimension.
            fdim (int): Encoded feature dimension.
            pool (str): MIL aggregation method: collate, attn, mean, None.
            dropout (int): Dropout rate.
        """
        super().__init__(fdim, local=True if pool == "collate" else False)

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
            nn.Dropout(p=dropout),
            torch.nn.Linear(hdim, fdim),
            nn.ReLU(True),
        )
        if pool == "collate":
            self.aggregate = self.collate_graphs
        elif pool == "attn":
            self.aggregate = GraphAttentionPool(fdim=fdim, hdim=fdim, dropout=dropout)
        elif pool == "mean":
            self.aggregate = MeanAggregation()
        elif pool is None:
            self.aggregate = None
        else:
            raise NotImplementedError(f"Aggregation method {pool} not implemented.")

    def normalize_graphs(self, data: Batch) -> Batch:
        """Normalize graph features and edge attributes as described in the paper."""

        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        data.x = data.x.type(torch.FloatTensor).to(data.x.device)
        data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(
            data.edge_attr.device
        )
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    @staticmethod
    def collate_graphs(
        x: torch.Tensor, index: torch.Tensor, dim_size: int = None
    ) -> torch.Tensor:
        """
        Converts graph embeddings of shape (n_graphs, fdim) to tensor of
        shape (batch_size, max(graphs_per_pat), fdim), with 0 padding in dim 1.

        Args:
            x (torch.Tensor): Graph embeddings.
            index (torch.Tensor): Batch indices, ie which patient each graph belongs to.
            dim_size (int): Size of the batch dimension (ie number of patients).
        """
        dim_size = index[-1].item() + 1 if dim_size is None else dim_size
        x = x.squeeze(1)
        graphs = [x[index == i] for i in range(dim_size)]
        padded = pad_sequence(graphs, batch_first=True, padding_value=0)
        return padded

    def get_latents(self, **kwargs: torch.Tensor) -> tuple:
        data, pat_idxs = kwargs["x_graph"]
        # data.batch records the indices of individual graphs
        # pat_idxs maps these back to patients for MIL aggregation
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
            x = self.aggregate(x, index=pat_idxs, dim_size=pat_idxs[-1].item() + 1)
        return x

    def forward(self, **kwargs: torch.Tensor) -> tuple:
        x = self.get_latents(**kwargs)
        if self.local:
            mask = torch.any(x != 0, dim=-1)
            return self.predict(x, mask)
        return self.predict(x)

    def freeze(self, freeze: bool) -> None:
        dfs_freeze(self, freeze)

    def get_attn_score(self, single_graph) -> tuple:
        data = single_graph
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
            A = self.aggregate.attn(x)
        return A


# --- Path ---
class ResNetClassifier(BaseEncoder):
    def __init__(self, xdim: int = 2048, pool: str = None) -> None:
        """
        Skeleton classifier for pre-extracted resnet features.

        Args:
            xdim (int): Input feature dimension.
            pool (str): MIL aggregation method: attn, mean, None.
        """
        super().__init__(xdim)

        if pool == "attn":
            self.aggregate = MaskedAttentionPool(fdim=xdim, hdim=xdim // 2, dropout=0)
        elif pool == "mean":
            self.aggregate = MaskedMeanPool()
        elif pool is None:
            self.aggregate = None
        else:
            raise NotImplementedError(f"Aggregation method {pool} not implemented.")

    def forward(self, **kwargs: torch.Tensor) -> tuple:
        x = kwargs["x_path"]
        x = self.aggregate(x) if self.aggregate else x
        return self.predict(x)


class VGGNet(BaseEncoder):
    def __init__(self, vgg_layers: nn.Module, fdim: int = 32) -> None:
        """
        VGG19 feature extractor for pathology images.

        Args:
            vgg_layers (nn.Module): Pretrained VGG19 convolutional layers.
            fdim (int): Encoded feature dimension.
        """
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

        dfs_freeze(self.conv_layers, True)

    def get_latents(self, **kwargs: torch.Tensor) -> tuple:
        x = kwargs["x_path"]
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, **kwargs: torch.Tensor) -> tuple:
        x = self.get_latents(**kwargs)
        return self.predict(x)


def make_layers(cfg: list) -> nn.Module:
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


def build_vgg19_encoder(fdim: int = 32) -> nn.Module:
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M"]
    cfg += [512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    model = VGGNet(make_layers(cfg), fdim=fdim)

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
