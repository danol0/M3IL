import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv, SAGPooling
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


# --- Utility ---
def define_model(opt, k):
    omic_dim = 320 if opt.use_rna else 80
    if opt.model == "omic":
        model = FFN(input_dim=omic_dim, omic_dim=32, dropout=opt.dropout)
    elif opt.model == "pathomic_qbt":
        model = QBTNet(
            k,
            task=opt.task,
            device=opt.device,
            feature_dim=32,
            n_queries=16,
            batch_size=opt.batch_size,
            transformer_layers=12,
            dropout=opt.dropout,
            omic_xdim=omic_dim,
        )
    elif opt.model == "graph":
        model = GraphNet(
            features=1036, nhid=128, graph_dim=32, dropout_rate=opt.dropout
        )
    else:
        raise NotImplementedError(f"Model {opt.model} not implemented")
    return model


def dfs_freeze(model, freeze=True):
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = not freeze
        dfs_freeze(child)


# --- Omics ---
class FFN(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout=0.25):
        super().__init__()
        hidden = [64, 48, 32, omic_dim] if input_dim == 80 else [128, 64, 48, omic_dim]

        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden[i - 1], hidden[i]))
            layers.append(nn.ELU())
            layers.append(nn.AlphaDropout(p=dropout, inplace=False))

        self.encoder = nn.Sequential(*layers)
        self.grade_clf = nn.Sequential(nn.Linear(omic_dim, 3), nn.LogSoftmax(dim=1))
        self.survival_clf = nn.Sequential(nn.Linear(omic_dim, 1), nn.Sigmoid())

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

    def forward(self, **kwargs):
        x = kwargs["x_omic"]
        features = self.encoder(x)
        grade = self.grade_clf(features)
        survival = self.survival_clf(features) * self.output_range + self.output_shift

        return features, grade, survival

    def freeze(self, freeze=True):
        dfs_freeze(self, freeze)


# --- Graph ---
class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        data.x = data.x.type(torch.FloatTensor)
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class GraphNet(torch.nn.Module):
    def __init__(self, features=1036, nhid=128, graph_dim=32, dropout_rate=0.25):
        super().__init__()

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
        self.survival_clf = nn.Sequential(nn.Linear(graph_dim, 1), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        data, graphs_per_pat = kwargs["x_graph"]
        data = NormalizeFeaturesV2()(data)
        data = NormalizeEdgesV2()(data)
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
        # We want to pool the graph features belonging to the same patient
        patient_indices = torch.arange(len(graphs_per_pat))
        batch_vector = torch.repeat_interleave(patient_indices, graphs_per_pat)

        # Now we can use the batch vector to pool the graphs
        x = gap(x, batch_vector)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate)

        # The result is a single feature vector for each patient that aggregates all the graphs
        features = F.relu(self.lin2(x))

        grade = self.grade_clf(features)
        survival = self.survival_clf(features) * self.output_range + self.output_shift

        return features, grade, survival


# def aggregrate(batch_vector, tensor, agg_type="mean"):
#     unique_patients = torch.unique(batch_vector)
#     if agg_type == "sum":
#         return torch.stack([tensor[batch_vector == patient].sum(dim=0) for patient in unique_patients])
#     elif agg_type == "mean":
#         return torch.stack([tensor[batch_vector == patient].mean(dim=0) for patient in unique_patients])
#     elif agg_type == "max":
#         return torch.stack([tensor[batch_vector == patient].max(dim=0).values for patient in unique_patients])
#     else:
#         raise ValueError(f"Unknown aggregation type {agg_type}")


# --- QBT ---
class QBTNet(nn.Module):
    def __init__(
        self,
        k,
        task,
        device,
        feature_dim=32,
        n_queries=16,
        batch_size=32,
        transformer_layers=12,
        dropout=0.25,
        omic_xdim=80,
    ):
        super().__init__()
        query_dim = feature_dim
        n_heads = 4

        self.n = transformer_layers
        self.batch_size = batch_size

        self.omic_net = FFN(input_dim=omic_xdim, omic_dim=feature_dim, dropout=dropout)
        best_omic_ckpt = torch.load(f"checkpoints/{task}/omic/omic_{k}.pt")
        self.omic_net.load_state_dict(best_omic_ckpt["model"])
        self.omic_net = self.omic_net.to(device)

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
        self.survival_clf = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

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
        survival = self.survival_clf(features) * self.output_range + self.output_shift
        return features, grade, survival

        # TODO: compare these differences
        # grade = self.LSM(self.grade_clf(qs).mean(dim=1))
        # survival = (
        #     self.sigmoid(self.survival_clf(qs).mean(dim=1)) * self.output_range
        #     + self.output_shift
        # )
        # return 0, grade, survival
