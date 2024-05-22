import torch
import torch.nn as nn


def define_fusion(opt):
    """Defines tensor fusion as described in the paper."""

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
        return Bimodal(
            device=opt.device,
            gate1=gate_path,
            gate2=gate_omic,
            feature_dim=32,
            scale_dim1=path_scale,
            scale_dim2=omic_scale,
            dropout=opt.dropout,
        )
    elif opt.model == "graphomic":
        return Bimodal(
            device=opt.device,
            gate1=gate_graph,
            gate2=gate_omic,
            feature_dim=32,
            scale_dim1=graph_scale,
            scale_dim2=omic_scale,
            dropout=opt.dropout,
        )
    elif opt.model == "pathgraphomic":
        return Trimodal(
            device=opt.device,
            gate_graph_with_omic=1 if opt.task == "surv" else 0,
            gate_path=gate_path,
            gate_graph=gate_graph,
            gate_omic=gate_omic,
            feature_dim=32,
            path_scale=path_scale,
            graph_scale=graph_scale,
            omic_scale=omic_scale,
            dropout=0.25,
        )
    else:
        raise NotImplementedError(f"Fusion for {opt.model} not implemented")


class TensorFusion(nn.Module):
    def __init__(self, device, feature_dim, dropout=0.25):
        """Base class for bimodal and trimodal tensor fusion."""

        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()

    def _tensor_fusion(self, tensors):
        fused = tensors[0]
        for t in tensors[1:]:
            fused = torch.bmm(fused.unsqueeze(2), t.unsqueeze(1)).flatten(start_dim=1)
        return fused

    def _create_gate_layers(self, feature_dim, scaled_dim, dropout):
        rescale_layer = nn.Sequential(nn.Linear(feature_dim, scaled_dim), nn.ReLU())
        gate_weight_layer = nn.Bilinear(feature_dim, feature_dim, scaled_dim)
        out_layer = nn.Sequential(
            nn.Linear(scaled_dim, scaled_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )
        return rescale_layer, gate_weight_layer, out_layer

    def _rescale_and_gate(self, x, x_gate, rescale_layer, gate_layer, out_layer, gate):
        """NOTE: Gating behaviour has been changed from the paper.
        Specifically, the paper does not apply the rescaling layer if the gate is off.
        This would result in an error if a mode is rescaled but not gated, as the out layer
        would expect the rescaled input. This implementation applies the rescaling layer
        regardless of the gate state."""

        o = rescale_layer(x)
        w = self.sigmoid(gate_layer(x, x_gate)) if gate else 1
        o = out_layer(w * o)
        return o

    def _append_one(self, o):
        return torch.cat(
            (o, torch.FloatTensor(o.shape[0], 1).fill_(1).to(self.device)), 1
        )


class Trimodal(TensorFusion):
    def __init__(
        self,
        device,
        gate_graph_with_omic=1,
        gate_path=1,
        gate_graph=1,
        gate_omic=1,
        feature_dim=32,
        path_scale=1,
        graph_scale=1,
        omic_scale=1,
        dropout=0.25,
    ):
        super().__init__(device, feature_dim, dropout)
        mmhid = 64
        self.gate_path = gate_path
        self.gate_graph = gate_graph
        self.gate_omic = gate_omic
        self.device = device
        self.gate_graph_with_omic = gate_graph_with_omic

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

        self.post_fusion_dropout = nn.Dropout(p=dropout)
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
            x_omic if self.gate_graph_with_omic else x_path,
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
        fused = self._tensor_fusion([o1, o2, o3])
        out = self.post_fusion_dropout(fused)
        out = self.encoder(out)
        return out


class Bimodal(TensorFusion):
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
        super().__init__(device, feature_dim, dropout)
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
        self.encoder = nn.Sequential(
            nn.Linear((dim1_scaled + 1) * (dim2_scaled + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mmhid, mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

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
        fused = self._tensor_fusion([o1, o2])
        out = self.post_fusion_dropout(fused)
        out = self.encoder(out)
        return out
