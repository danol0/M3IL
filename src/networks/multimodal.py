import torch
from torch import nn
from torch.nn import Parameter

from src.networks.fusion import define_tensor_fusion
from src.networks.unimodal import (
    FFN,
    GNN,
    BaseEncoder,
    MaskedAttentionPool,
    MaskedMeanPool,
    get_vgg19,
)


def print_load(ckpt, device):
    print(f"Loading {ckpt}")
    try:
        return torch.load(ckpt, map_location=device)
    except FileNotFoundError:
        print(f"Checkpoint {ckpt} not found. Exiting...")
        exit(1)


class DynamicFusion(BaseEncoder):
    def __init__(self, opt, mmfdim):
        """Dynamic multimodal fusion of path, graph and omics data:
        1. Encoding of each modality
        2. Optional MIL instance aggregation of path vectors
        3. Modality fusion to produce a single feature vector"""
        super().__init__(mmfdim)

        # --- Feature encoders ---
        if "omic" in opt.model:
            self.omic_net = FFN(opt)
            rna = "_rna" if opt.rna else ""
            ckpt = print_load(
                f"checkpoints/{opt.task}/omic{rna}/omic_{opt.k}.pt", device=opt.device
            )
            self.omic_net.load_state_dict(ckpt["model"])
            self.omic_net = self.omic_net.to(opt.device)
            self.omic_net.freeze(True)

        if "graph" in opt.model:
            self.graph_net = GNN(opt)
            attn = "_attn" if opt.attn_pool or "qbt" not in opt.model else ""
            mil = (
                "instance"
                if opt.mil in ("instance", "paper") or "qbt" in opt.model
                else opt.mil
            )
            graph_ckpt = print_load(
                f"checkpoints/{opt.task}/graph_{mil}{attn}/graph_{opt.k}.pt",
                device=opt.device,
            )
            self.graph_net.load_state_dict(graph_ckpt["model"])
            self.graph_net = self.graph_net.to(opt.device)
            self.graph_net.freeze(True)

        if "path" in opt.model:
            self.aggregate = nn.Identity()
            if opt.mil == "pat":
                self.aggregate = (
                    MaskedAttentionPool(features=32, hidden=32, dropout=0)
                    if opt.attn_pool
                    else MaskedMeanPool()
                )
            if not opt.use_vgg:
                self.path_net = get_vgg19(opt)
                path_ckpt = print_load(
                    f"checkpoints/{opt.task}/path_instance/path_{opt.k}.pt",
                    device=opt.device,
                )
                self.path_net.load_state_dict(path_ckpt["model"])
                self.path_net = self.path_net.to(opt.device)
                self.aggregate = nn.Identity()
                self.path_net.freeze(True)  # Remains frozen

        # --- Fusion ---
        if "qbt" in opt.model:
            assert opt.mil == "pat", "QBT only supports pat level MIL."
            self.fusion = QBT(opt)
        else:
            self.fusion = define_tensor_fusion(opt, mmfdim=mmfdim)

    def forward(self, **kwargs):
        p, g, o = None, None, None

        if hasattr(self, "path_net"):
            p = self.path_net(x_path=kwargs["x_path"])[0]
            p = self.aggregate(p)
        elif hasattr(self, "aggregate"):
            p = self.aggregate(kwargs["x_path"])

        if hasattr(self, "graph_net"):
            g = self.graph_net(x_graph=kwargs["x_graph"])[0]

        if hasattr(self, "omic_net"):
            o = self.omic_net(x_omic=kwargs["x_omic"])[0]

        x = self.fusion(f_omic=o, f_graph=g, f_path=p)
        return super().forward(x)


class QBT(nn.Module):
    def __init__(
        self,
        opt,
        fdim=32,
        n_queries=16,
        n_heads=4,
        transformer_layers=1,
    ):
        super().__init__()
        self.n = transformer_layers

        if "graph" in opt.model:
            self.graph_cross_attn = [
                nn.MultiheadAttention(
                    fdim, n_heads, dropout=opt.dropout, batch_first=True
                )
                for _ in range(self.n)
            ]

        if "path" in opt.model:
            self.LN = nn.LayerNorm(fdim)
            self.correlation = nn.MultiheadAttention(
                fdim, n_heads, dropout=opt.dropout, batch_first=True
            )
            self.path_cross_attn = [
                nn.MultiheadAttention(
                    fdim, n_heads, dropout=opt.dropout, batch_first=True
                )
                for _ in range(self.n)
            ]

        if "omic" in opt.model:
            self.omics_cross_attn = [
                nn.MultiheadAttention(
                    fdim, n_heads, dropout=opt.dropout, batch_first=True
                )
                for _ in range(self.n)
            ]

        self.FFN = [
            nn.Sequential(
                nn.Linear(fdim, fdim),
                nn.Dropout(opt.dropout),
                nn.ReLU(),
                nn.Linear(fdim, fdim),
                nn.ReLU(),
            )
            for _ in range(self.n)
        ]

        self.Qs = Parameter(torch.randn(n_queries, fdim), requires_grad=True)

    def forward(self, **kwargs):
        o, g, p = (
            kwargs.get("f_omic", None),
            kwargs.get("f_graph", None),
            kwargs.get("f_path", None),
        )
        o = o.unsqueeze(1) if o is not None else None
        p = p.unsqueeze(1) if p is not None else None

        batch_size = (
            o.size(0) if o is not None else g.size(0) if g is not None else p.size(0)
        )

        Q = self.Qs.repeat(batch_size, 1, 1)  # batch x 1 x fdim

        if p is not None:
            p = self.LN(p)
            p = p + self.correlation(p, p, p)[0]

        for i in range(self.n):
            Q = self.omics_cross_attn[i](Q, o, o)[0] if o is not None else Q
            Q = self.path_cross_attn[i](Q, p, p)[0] if p is not None else Q
            Q = self.graph_cross_attn[i](Q, g, g)[0] if g is not None else Q
            Q = self.FFN[i](Q)

        return Q.mean(dim=1)
