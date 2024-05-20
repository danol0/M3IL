import torch
import torch.nn as nn


def define_fusion(opt):
    """Defines bimodal fusion as described in the paper."""

    gate_path = (
        0 if opt.task == "grad" and opt.model in ("pathomic", "pathgraphomic") else 1
    )
    gate_graph = 0 if opt.task == "grad" and opt.model == "graphomic" else 1
    gate_omic = (
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
            gate1=gate_path,
            gate2=gate_omic,
            feature_dim=32,
            scale_dim1=path_scale,
            scale_dim2=omic_scale,
            dropout_rate=opt.dropout,
        )
    elif opt.model == "graphomic":
        return BilinearFusion(
            device=opt.device,
            gate1=gate_graph,
            gate2=gate_omic,
            feature_dim=32,
            scale_dim1=graph_scale,
            scale_dim2=omic_scale,
            dropout_rate=opt.dropout,
        )
    elif opt.model == "pathgraphomic":
        return TrilinearFusion(
            device=opt.device,
            type_A=1 if opt.task == "surv" else 0,
            gate_path=gate_path,
            gate_graph=gate_graph,
            gate_omic=gate_omic,
            path_scale=path_scale,
            graph_scale=graph_scale,
            omic_scale=omic_scale,
            dropout=0.25,
        )
    else:
        raise NotImplementedError(f"Fusion for {opt.model} not implemented")


class TrilinearFusion(nn.Module):
    def __init__(
        self,
        device,
        type_A=1,
        gate_path=1,
        gate_graph=1,
        gate_omic=1,
        path_scale=1,
        graph_scale=1,
        omic_scale=1,
        dropout=0.25,
    ):
        super().__init__()
        mmhid = 96
        feature_dim = 32
        self.gate_path = gate_path
        self.gate_graph = gate_graph
        self.gate_omic = gate_omic
        self.device = device
        self.type_A = type_A

        path_scaled, graph_scaled, omic_scaled = (
            feature_dim // path_scale,
            feature_dim // graph_scale,
            feature_dim // omic_scale,
        )

        self.path_rescale, self.path_gate_weight, self.path_gated = (
            self._create_gate_layers(feature_dim, path_scaled, dropout)
        )
        self.graph_rescale, self.graph_gate_weight, self.graph_gated = (
            self._create_gate_layers(feature_dim, graph_scaled, dropout)
        )
        self.omic_rescale, self.omic_gate_weight, self.omic_gated = (
            self._create_gate_layers(feature_dim, omic_scaled, dropout)
        )

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder = nn.Sequential(
            nn.Linear(
                (path_scaled + 1) * (graph_scaled + 1) * (omic_scaled + 1), mmhid
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mmhid, mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def _create_gate_layers(self, feature_dim, scaled_dim, dropout):
        rescale_layer = nn.Sequential(nn.Linear(feature_dim, scaled_dim), nn.ReLU())
        gate_weight_layer = nn.Sequential(
            nn.Bilinear(feature_dim, feature_dim, scaled_dim), nn.Sigmoid()
        )
        out_layer = nn.Sequential(
            nn.Linear(scaled_dim, scaled_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )
        return rescale_layer, gate_weight_layer, out_layer

    def forward(self, x_path, x_graph, x_omic):
        o1 = self._rescale_and_gate(
            x_path,
            x_omic,
            self.path_rescale,
            self.path_gate_weight,
            self.path_gated,
            self.gate_path,
        )
        o2 = self._rescale_and_gate(
            x_graph,
            x_omic if self.type_A else x_path,
            self.graph_rescale,
            self.graph_gate_weight,
            self.graph_gated,
            self.gate_graph,
        )
        o3 = self._rescale_and_gate(
            x_path,
            x_omic,
            self.omic_rescale,
            self.omic_gate_weight,
            self.omic_gated,
            self.gate_omic,
        )

        # Fusion
        o1 = self._append_one(o1)
        o2 = self._append_one(o2)
        o3 = self._append_one(o3)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder(out)
        return out

    def _rescale_and_gate(self, x, x_gate, rescale_layer, gate_layer, out_layer, gate):
        o = rescale_layer(x)
        w = gate_layer(x, x_gate) if gate else 1
        o = out_layer(w * o)
        return o

    def _append_one(self, o):
        return torch.cat(
            (o, torch.FloatTensor(o.shape[0], 1).fill_(1).to(self.device)), 1
        )


class BilinearFusion(nn.Module):
    def __init__(
        self,
        device,
        gate1=1,
        gate2=1,
        feature_dim=32,
        scale_dim1=1,
        scale_dim2=1,
        dropout=0.25,
    ):
        super(BilinearFusion, self).__init__()
        self.gate1 = gate1
        self.gate2 = gate2
        self.device = device
        mmhid = 64

        dim1_scaled, dim2_scaled = feature_dim // scale_dim1, feature_dim // scale_dim2

        self.rescale_1, self.gate_weight_1, self.out_1 = self._create_gate_layers(
            feature_dim, dim1_scaled, dropout
        )
        self.rescale_2, self.gate_weight_2, self.out_2 = self._create_gate_layers(
            feature_dim, dim2_scaled, dropout
        )

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.encoder1 = nn.Sequential(
            nn.Linear((feature_dim + 1) * (feature_dim + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid, mmhid), nn.ReLU(), nn.Dropout(p=dropout)
        )

    def _create_gate_layers(self, feature_dim, scaled_dim, dropout):
        rescale_layer = nn.Sequential(nn.Linear(feature_dim, scaled_dim), nn.ReLU())
        gate_weight_layer = nn.Sequential(
            nn.Bilinear(feature_dim, feature_dim, scaled_dim), nn.Sigmoid()
        )
        out_layer = nn.Sequential(
            nn.Linear(scaled_dim, scaled_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )
        return rescale_layer, gate_weight_layer, out_layer

    def forward(self, vec1, vec2):
        o1 = self._rescale_and_gate(
            vec1,
            vec2,
            self.rescale_1,
            self.gate_weight_1,
            self.out_1,
            self.gate1,
        )

        o2 = self._rescale_and_gate(
            vec2,
            vec1,
            self.rescale_2,
            self.gate_weight_2,
            self.out_2,
            self.gate2,
        )

        # Fusion
        o1 = self._append_one(o1)
        o2 = self._append_one(o2)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        out = self.encoder2(out)

        return out

    def _rescale_and_gate(self, x, x_gate, rescale_layer, gate_layer, out_layer, gate):
        o = rescale_layer(x)
        w = gate_layer(x, x_gate) if gate else 1
        o = out_layer(w * o)
        return o

    def _append_one(self, o):
        return torch.cat(
            (o, torch.FloatTensor(o.shape[0], 1).fill_(1).to(self.device)), 1
        )
