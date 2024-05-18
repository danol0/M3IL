import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv, SAGPooling
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from src.fusion import define_bifusion


# --- Utility ---
def define_model(opt):
    omic_dim = 320 if opt.rna else 80
    if opt.model == "omic":
        model = FFN(input_dim=omic_dim, omic_dim=32, dropout=opt.dropout)
    elif opt.model == "pathomic_qbt":
        model = QBTNet(
            opt,
            feature_dim=32,
            n_queries=16,
            transformer_layers=12,
            omic_xdim=omic_dim,
        )
    elif opt.model == "graph":
        model = GraphNet(dropout_rate=opt.dropout)
    elif opt.model == "pathomic":
        model = PathomicNet(opt, omic_xdim=omic_dim, feature_dim=32)
    elif opt.model == "graphomic":
        model = GraphomicNet(opt, omic_xdim=omic_dim)
    else:
        raise NotImplementedError(f"Model {opt.model} not implemented")
    return model


def dfs_freeze(model, freeze=True):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = not freeze
        dfs_freeze(child)


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

    def freeze(self, freeze=True):
        dfs_freeze(self, freeze)


# --- Graph ---
class NormalizeGraphs(object):
    r"""Column-normalises nodes and edges to sum to 1."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        data.x = data.x.type(torch.FloatTensor)
        data.edge_attr = data.edge_attr.type(torch.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class GraphNet(torch.nn.Module):
    def __init__(self, dropout_rate=0.25):
        super().__init__()

        features = 1036
        nhid = 128
        graph_dim = 32

        hidden = [features, nhid, nhid]

        pooling_ratio = 0.2
        self.dropout_rate = dropout_rate

        self.convs = torch.nn.ModuleList(
            [SAGEConv(in_channels, nhid) for in_channels in hidden]
        )
        self.pools = torch.nn.ModuleList(
            [SAGPooling(nhid, ratio=pooling_ratio) for _ in range(len(hidden))]
        )

        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        self.lin2 = torch.nn.Linear(nhid, graph_dim)

        self.grade_clf = nn.Sequential(nn.Linear(graph_dim, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(graph_dim, 1), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        data, graphs_per_pat = kwargs["x_graph"]
        data = NormalizeGraphs()(data)
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

        # x is now a tensor of shape [num_graphs, 2 * nhid]
        # For bagged MIL, each graph is itself a batch of graphs belonging to the patient
        # We want to pool the graph features belonging to the same patient
        patient_indices = torch.arange(len(graphs_per_pat))
        batch_vector = torch.repeat_interleave(patient_indices, graphs_per_pat)

        # Now we can use the batch vector to pool the graphs
        # For instance level MIL, this will do nothing
        x = gap(x, batch_vector)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate)

        # The result is a single feature vector for each patient that aggregates all the graphs
        features = F.relu(self.lin2(x))

        grade = self.grade_clf(features)
        hazard = self.hazard_clf(features) * self.output_range + self.output_shift

        return features, grade, hazard

    def freeze(self, freeze=True):
        dfs_freeze(self, freeze)


# --- FUSION ---
class PathomicNet(nn.Module):
    def __init__(self, opt, omic_xdim=80, feature_dim=32):
        super().__init__()
        mmhid = 64
        dropout = opt.dropout
        # TODO: experiment with reduced dropout layers
        self.omic_net = FFN(
            input_dim=omic_xdim, omic_dim=feature_dim, dropout=dropout, dropout_layer=0
        )
        rna = "_rna" if opt.rna else ""
        ckpt = torch.load(f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt")
        self.omic_net.load_state_dict(ckpt["model"])
        self.omic_net = self.omic_net.to(opt.device)

        # TODO: learned/flexible aggregation
        self.bagged = 1 if opt.mil == "pat" else 0

        self.fusion = define_bifusion(opt)
        self.grade_clf = nn.Sequential(nn.Linear(mmhid, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(mmhid, 1), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        self.omic_net.freeze(True)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        if self.bagged:
            path_vec = path_vec.sum(dim=1)
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
        self.grph_net = GraphNet(dropout_rate=opt.dropout)
        self.omic_net = FFN(
            input_dim=omic_xdim, omic_dim=feature_dim, dropout=dropout, dropout_layer=0
        )

        rna = "_rna" if opt.rna else ""
        omic_ckpt = torch.load(f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt")
        self.omic_net.load_state_dict(omic_ckpt["model"])
        self.omic_net = self.omic_net.to(opt.device)
        grph_ckpt = torch.load(f"checkpoints/{opt.task}/graph/graph_{opt.k}.pt")
        self.grph_net.load_state_dict(grph_ckpt["model"])
        self.grph_net = self.grph_net.to(opt.device)

        self.fusion = define_bifusion(opt)
        self.grade_clf = nn.Sequential(nn.Linear(feature_dim, 3), nn.LogSoftmax(dim=1))
        self.hazard_clf = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        self.omic_net.freeze(True)
        self.grph_net.freeze(True)

    def forward(self, **kwargs):
        grph_vec, _ = self.grph_net(x_grph=kwargs["x_grph"])
        omic_vec, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(grph_vec, omic_vec)
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
        ckpt = torch.load(f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt")
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

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

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
