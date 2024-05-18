import torch
import torch.nn as nn


def define_bifusion(opt):
    """Defines bimodal fusion as described in the paper."""

    path_gate = (
        0 if opt.task == "grad" and opt.model in ("pathomic", "pathgraphomic") else 1
    )
    graph_gate = 0 if opt.task == "grad" and opt.model == "graphomic" else 1
    omic_gate = (
        0 if opt.task == "surv" and opt.model in ("pathomic", "graphomic") else 1
    )
    path_scale = 1
    graph_scale = (
        2 if opt.task == "surv" and opt.model in ("pathgraphomic", "graphomic") else 1
    )
    omic_scale = (
        2 if opt.task == "grad" and opt.model in ("pathomic", "graphomic") else 1
    )

    if opt.model == "pathomic":
        return BilinearFusion(
            device=opt.device,
            gate1=path_gate,
            gate2=omic_gate,
            feature_dim=32,
            scale_dim1=path_scale,
            scale_dim2=omic_scale,
            dropout_rate=opt.dropout,
        )
    elif opt.model == "graphomic":
        return BilinearFusion(
            device=opt.device,
            gate1=graph_gate,
            gate2=omic_gate,
            feature_dim=32,
            scale_dim1=graph_scale,
            scale_dim2=omic_scale,
            dropout_rate=opt.dropout,
        )
    else:
        raise NotImplementedError(f"Fusion for {opt.model} not implemented")


class BilinearFusion(nn.Module):
    def __init__(
        self,
        device,
        gate1=1,
        gate2=1,
        feature_dim=32,
        scale_dim1=1,
        scale_dim2=1,
        dropout_rate=0.25,
    ):
        super(BilinearFusion, self).__init__()
        self.gate1 = gate1
        self.gate2 = gate2
        self.device = device
        dim1, dim2 = feature_dim, feature_dim
        mmhid = 64

        dim1_og, dim2_og, dim1, dim2 = (
            dim1,
            dim2,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
        )

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1)
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2)
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

    def forward(self, vec1, vec2):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2)
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2)
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        # Fusion
        o1 = torch.cat(
            (o1, torch.FloatTensor(o1.shape[0], 1).fill_(1).to(self.device)), 1
        )
        o2 = torch.cat(
            (o2, torch.FloatTensor(o2.shape[0], 1).fill_(1).to(self.device)), 1
        )
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(
            start_dim=1
        )  # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        out = self.encoder2(out)

        return out
