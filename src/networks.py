import torch
import torch.nn as nn
from torch.nn import Parameter


# --- Utility ---
def define_model(opt, k):
    omic_dim = 320 if opt.use_rna else 80
    if opt.model == "omic":
        model = FFN(input_dim=omic_dim, omic_dim=32, dropout=opt.dropout)
    elif opt.model == "qbt":
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
    else:
        raise NotImplementedError(f"Model {opt.model} not implemented")
    return model


def dfs_freeze(model, freeze=True):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = not freeze
        dfs_freeze(child)


# --- Omics ---
class FFN(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout=0.25):
        super().__init__()
        hidden = [64, 48, 32, omic_dim]

        layers = []
        for i in range(len(hidden)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden[i - 1], hidden[i]))
            layers.append(nn.ELU())
            layers.append(nn.AlphaDropout(p=dropout, inplace=False))

        self.encoder = nn.Sequential(*layers)
        self.grade_clf = nn.Linear(omic_dim, 3)
        self.survival_clf = nn.Linear(omic_dim, 1)

        self.register_buffer("output_range", torch.FloatTensor([6]))
        self.register_buffer("output_shift", torch.FloatTensor([-3]))

        self.LSM = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):
        x = kwargs["x_omic"]
        features = self.encoder(x)
        grade = self.LSM(self.grade_clf(features))
        survival = (
            self.sigmoid(self.survival_clf(features)) * self.output_range
            + self.output_shift
        )

        return features, grade, survival

    def freeze(self, freeze=True):
        dfs_freeze(self, freeze)


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

        self.grade_clf = nn.Linear(feature_dim, 3)
        self.survival_clf = nn.Linear(feature_dim, 1)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        self.LSM = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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

        # out = self.output_layer(qs).mean(dim=1)
        grade = self.LSM(self.grade_clf(qs).mean(dim=1))
        survival = (
            self.sigmoid(self.survival_clf(qs).mean(dim=1)) * self.output_range
            + self.output_shift
        )

        return 0, grade, survival
