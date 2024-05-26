import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import Parameter
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, SAGPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter, softmax

from src.fusion import define_fusion
from src.utils import print_load


# --- Utility ---
def define_model(opt):
    omic_dim = 320 if opt.rna else 80
    if opt.model == "omic":
        model = FFN(input_dim=omic_dim, omic_dim=32, dropout=opt.dropout)
    elif opt.model == "graph":
        model = GraphNet(dropout=opt.dropout, attn_pool=opt.attn_pool)
    elif opt.model == "path":
        model = get_vgg19(opt)
    elif opt.model == "pathomic":
        model = PathomicNet(opt, omic_xdim=omic_dim)
    elif opt.model == "graphomic":
        model = GraphomicNet(opt, omic_xdim=omic_dim)
    elif opt.model == "pathgraphomic":
        model = PathgraphomicNet(opt, omic_xdim=omic_dim)
    elif opt.model == "pathomic_qbt":
        model = QBTNet(
            opt,
            feature_dim=32,
            n_queries=16,
            transformer_layers=12,
            omic_xdim=omic_dim,
        )
    else:
        raise NotImplementedError(f"Model {opt.model} not implemented")
    return model


def dfs_freeze(model, freeze):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = not freeze
        dfs_freeze(child, freeze)


# --- MIL ---


class AttentionAggregation(nn.Module):
    """Source: https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py"""

    def __init__(self, features=32, hidden=32, dropout=0.25):
        super().__init__()
        self.a = nn.Sequential(
            nn.Linear(features, hidden), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.b = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Dropout(dropout)
        )
        self.c = nn.Linear(hidden, 1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        A = a.mul(b)
        A = self.c(A)
        pad = torch.all(x == 0, dim=-1)
        A[pad] = -float("Inf")
        A = F.softmax(A, dim=1)
        return torch.sum(A * x, dim=1)


class GlobalAttentionPool(torch.nn.Module):
    """Attention pooling for graphs.
    Source: https://pytorch-geometric.readthedocs.io/en/1.3.1/_modules/torch_geometric/nn/glob/attention.html
    """

    def __init__(self, features=32, hidden=16, dropout=0.25):
        super().__init__()
        self.a = nn.Sequential(
            nn.Linear(features, hidden), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.b = nn.Sequential(
            nn.Linear(features, hidden), nn.Tanh(), nn.Dropout(dropout)
        )
        self.c = nn.Linear(hidden, 1)

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


# --- Path ---
class PathNet(nn.Module):
    def __init__(self, vgg_layers, opt, path_dim=32):
        super().__init__()
        self.conv_layers = vgg_layers
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Avgpool for MPS compatibility
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05),
        )

        self.bagged = 1 if opt.mil == "pat" else 0
        if opt.attn_pool == 1:
            self.aggregate = AttentionAggregation(
                features=path_dim, hidden=16, dropout=0.25
            )
        else:
            self.aggregate = self.mean_aggregation

        self.grade_clf = nn.Sequential(nn.Linear(path_dim, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(path_dim, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

        dfs_freeze(self.conv_layers, freeze=True)

    @staticmethod
    def mean_aggregation(x):
        mask = torch.any(x != 0, dim=-1)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    def forward(self, **kwargs):
        x = kwargs["x_path"]
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        if self.bagged:
            features = self.aggregate(features)
        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift
        return features, grade, hazard


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


def get_vgg19(opt):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    model = PathNet(make_layers(cfg), opt, path_dim=32)

    pretrained_dict = load_state_dict_from_url(
        "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth", progress=True
    )

    for key in list(pretrained_dict.keys()):
        if "classifier" in key:
            pretrained_dict.pop(key)
        if key.startswith("features."):
            pretrained_dict[key[9:]] = pretrained_dict.pop(key)

    print("Initializing Path Weights")
    model.conv_layers.load_state_dict(pretrained_dict, strict=True)

    return model


# --- Omic ---
class FFN(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout=0.25, dropout_layer=0):
        super().__init__()
        hidden = [64, 48, 32, omic_dim]
        # if input_dim == 80 else [128, 64, 48, omic_dim]

        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden[i - 1], hidden[i]))
            layers.append(nn.ELU())
            if i >= dropout_layer:
                layers.append(nn.AlphaDropout(p=dropout, inplace=False))
            else:
                layers.append(nn.AlphaDropout(p=0, inplace=False))

        self.encoder = nn.Sequential(*layers)
        self.grade_clf = nn.Sequential(nn.Linear(omic_dim, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(omic_dim, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

    def forward(self, **kwargs):
        x = kwargs["x_omic"]
        features = self.encoder(x)
        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift
        return features, grade, hazard

    def freeze(self, freeze):
        # print("%sreezing omic net" % ("F" if freeze else "Unf"))
        dfs_freeze(self, freeze)


# --- Graph ---
class GraphNet(torch.nn.Module):
    def __init__(self, dropout=0.25, attn_pool=False):
        super().__init__()

        features = 1036
        nhid = 128
        graph_dim = 32

        hidden = [features, nhid, nhid]

        pooling_ratio = 0.2
        self.dropout = dropout

        self.convs = torch.nn.ModuleList(
            [SAGEConv(in_channels, nhid) for in_channels in hidden]
        )
        self.pools = torch.nn.ModuleList(
            [SAGPooling(nhid, ratio=pooling_ratio) for _ in range(len(hidden))]
        )

        self.encoder = nn.Sequential(
            torch.nn.Linear(nhid * 2, nhid),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            torch.nn.Linear(nhid, graph_dim),
            nn.ReLU(True),
        )

        self.aggregate = (
            GlobalAttentionPool(features=graph_dim, hidden=16, dropout=dropout)
            if attn_pool
            else MeanAggregation()
        )

        self.grade_clf = nn.Sequential(nn.Linear(graph_dim, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(graph_dim, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

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
            xs.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))

        x = torch.sum(torch.stack(xs), dim=0)
        x = self.encoder(x)
        features = self.aggregate(x, pat_idxs)
        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift
        return features, grade, hazard

    def freeze(self, freeze):
        # print("%sreezing graph net" % ("F" if freeze else "Unf"))
        dfs_freeze(self, freeze)

    def normalize_graphs(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        data.x = data.x.type(torch.FloatTensor).to(data.x.device)
        data.edge_attr = data.edge_attr.type(torch.FloatTensor).to(
            data.edge_attr.device
        )
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data


# --- FUSION ---
class PathomicNet(nn.Module):
    def __init__(self, opt, omic_xdim=80):
        super().__init__()
        feature_dim = 32
        mmhid = 64
        dropout = opt.dropout
        # TODO: experiment with reduced dropout layers
        self.omic_net = FFN(
            input_dim=omic_xdim, omic_dim=feature_dim, dropout=dropout, dropout_layer=0
        )
        rna = "_rna" if opt.rna else ""
        ckpt = print_load(f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt")
        self.omic_net.load_state_dict(ckpt["model"])
        self.omic_net = self.omic_net.to(opt.device)

        self.bagged = 1 if opt.mil == "pat" else 0
        if opt.attn_pool == 1:
            self.aggregate = AttentionAggregation(
                features=feature_dim, hidden=16, dropout=0.25
            )
        else:
            self.aggregate = self.mean_aggregation

        self.fusion = define_fusion(opt)
        self.grade_clf = nn.Sequential(nn.Linear(mmhid, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(mmhid, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

        self.omic_net.freeze(True)

    @staticmethod
    def mean_aggregation(x):
        mask = torch.any(x != 0, dim=-1)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        if self.bagged:
            path_vec = self.aggregate(path_vec)
        omic_vec, _, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(path_vec, omic_vec)
        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift
        return features, grade, hazard


class GraphomicNet(nn.Module):
    def __init__(self, opt, omic_xdim=80):
        super().__init__()
        dropout = opt.dropout
        feature_dim = 32
        mmhid = 64
        self.graph_net = GraphNet(dropout=opt.dropout, attn_pool=opt.attn_pool)
        self.omic_net = FFN(
            input_dim=omic_xdim, omic_dim=feature_dim, dropout=dropout, dropout_layer=0
        )

        rna = "_rna" if opt.rna else ""
        attn = "_attn" if opt.attn_pool else ""
        omic_ckpt = print_load(f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt")
        self.omic_net.load_state_dict(omic_ckpt["model"])
        self.omic_net = self.omic_net.to(opt.device)
        graph_ckpt = print_load(
            f"checkpoints/{opt.task}/graph_{opt.mil}{attn}/graph_{opt.k}.pt"
        )
        self.graph_net.load_state_dict(graph_ckpt["model"])
        self.graph_net = self.graph_net.to(opt.device)

        self.fusion = define_fusion(opt)
        self.grade_clf = nn.Sequential(nn.Linear(mmhid, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(mmhid, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

        self.omic_net.freeze(True)
        self.graph_net.freeze(True)

    def forward(self, **kwargs):
        graph_vec, _, _ = self.graph_net(x_graph=kwargs["x_graph"])
        omic_vec, _, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(graph_vec, omic_vec)
        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift
        return features, grade, hazard


class PathgraphomicNet(nn.Module):
    def __init__(self, opt, omic_xdim=80):
        super().__init__()
        dropout = opt.dropout
        feature_dim = 32
        mmhid = 64
        self.graph_net = GraphNet(dropout=opt.dropout, attn_pool=opt.attn_pool)
        self.omic_net = FFN(
            input_dim=omic_xdim, omic_dim=feature_dim, dropout=dropout, dropout_layer=0
        )

        self.bagged = 1 if opt.mil == "pat" else 0
        if opt.attn_pool == 1:
            self.aggregate = AttentionAggregation(
                features=feature_dim, hidden=16, dropout=0.25
            )
        else:
            self.aggregate = self.mean_aggregation

        rna = "_rna" if opt.rna else ""
        attn = "_attn" if opt.attn_pool else ""
        omic_ckpt = print_load(f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt")
        self.omic_net.load_state_dict(omic_ckpt["model"])
        self.omic_net = self.omic_net.to(opt.device)
        mil = "instance" if opt.mil in ("instance", "paper") else opt.mil
        graph_ckpt = print_load(
            f"checkpoints/{opt.task}/graph_{mil}{attn}/graph_{opt.k}.pt"
        )
        self.graph_net.load_state_dict(graph_ckpt["model"])
        self.graph_net = self.graph_net.to(opt.device)

        self.fusion = define_fusion(opt)
        self.grade_clf = nn.Sequential(nn.Linear(mmhid, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(mmhid, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

        self.omic_net.freeze(True)
        self.graph_net.freeze(True)

    @staticmethod
    def mean_aggregation(x):
        mask = torch.any(x != 0, dim=-1)
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        if self.bagged:
            path_vec = self.aggregate(path_vec)
        graph_vec, _, _ = self.graph_net(x_graph=kwargs["x_graph"])
        omic_vec, _, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(path_vec, graph_vec, omic_vec)
        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift

        return features, grade, hazard


# --- QBT ---
class QBTNet(nn.Module):
    def __init__(
        self,
        opt,
        feature_dim=32,
        n_queries=16,
        transformer_layers=12,
        omic_xdim=80,
    ):
        super().__init__()
        query_dim = feature_dim
        n_heads = 4
        dropout = opt.dropout

        self.n = transformer_layers
        self.batch_size = opt.batch_size

        self.omic_net = FFN(
            input_dim=omic_xdim, omic_dim=feature_dim, dropout=dropout, dropout_layer=2
        )
        rna = "_rna" if opt.rna else ""
        ckpt = print_load(f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt")
        self.omic_net.load_state_dict(ckpt["model"])
        self.omic_net = self.omic_net.to(opt.device)

        self.LN = nn.LayerNorm(feature_dim)
        self.correlation = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.query_omics_attention = [
            nn.MultiheadAttention(query_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(self.n)
        ]

        self.query_path_attention = [
            nn.MultiheadAttention(query_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(self.n)
        ]

        self.FFN = [
            nn.Sequential(
                nn.Linear(query_dim, query_dim),
                nn.ReLU(),
                nn.Linear(query_dim, query_dim),
            )
            for _ in range(self.n)
        ]

        # Define learnable queries for the attention module
        self.Qs = Parameter(torch.randn(n_queries, query_dim), requires_grad=True)

        self.grade_clf = nn.Sequential(nn.Linear(feature_dim, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

        self.omic_net.freeze(True)

    def forward(self, **kwargs):

        omic_embeddings = self.omic_net(**kwargs)[0].unsqueeze(1)  # batch x 1 x 32
        qs = self.Qs.repeat(omic_embeddings.size(0), 1, 1)  # batch x 1 x 32
        image_embeddings = kwargs["x_path"]
        img_e_norm = self.LN(image_embeddings)  # batch x 9 x 32
        image_embeddings = (
            image_embeddings + self.correlation(img_e_norm, img_e_norm, img_e_norm)[0]
        )

        for i in range(self.n):
            qs = self.query_omics_attention[i](qs, omic_embeddings, omic_embeddings)[0]
            qs = self.query_path_attention[i](qs, image_embeddings, image_embeddings)[0]
            qs = self.FFN[i](qs)

        features = qs.mean(dim=1)  # TODO: sum or mean
        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift
        return features, grade, hazard

        # TODO: compare these differences
        # grade = self.LSM(self.grade_clf(qs).mean(dim=1))
        # hazard = (
        #     self.sigmoid(self.hazard_clf(qs).mean(dim=1)) * self.output_range
        #     + self.output_shift
        # )
        # return 0, grade, hazard
