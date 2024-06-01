import torch
import torch.nn as nn


def define_tensor_fusion(opt, mmfdim=64):
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

    if opt.model in ("pathomic", "graphomic"):
        return Bimodal(
            device=opt.device,
            fdim=32,
            mmfdim=mmfdim,
            gate1=gate_path if opt.model == "pathomic" else gate_graph,
            gate2=gate_omic,
            scale_dim1=path_scale if opt.model == "pathomic" else graph_scale,
            scale_dim2=omic_scale,
            dropout=opt.dropout,
        )
    elif opt.model == "pathgraphomic":
        return Trimodal(
            device=opt.device,
            fdim=32,
            mmfdim=mmfdim,
            gate_graph_with_omic=1 if opt.task == "surv" else 0,
            gate_path=gate_path,
            gate_graph=gate_graph,
            gate_omic=gate_omic,
            path_scale=path_scale,
            graph_scale=graph_scale,
            omic_scale=omic_scale,
            dropout=opt.dropout,
        )
    else:
        raise NotImplementedError(f"Fusion for {opt.model} not implemented")


class TensorFusion(nn.Module):
    def __init__(self, device, fdim, dropout):
        """Base class for bimodal and trimodal tensor fusion."""

        super().__init__()
        self.device = device
        self.fdim = fdim
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()

    def _tensor_fusion(self, tensors):
        fused = tensors[0]
        for t in tensors[1:]:
            fused = torch.bmm(fused.unsqueeze(2), t.unsqueeze(1)).flatten(start_dim=1)
        return fused

    def _create_gate_layers(self, scaled_dim):
        rescale_layer = nn.Sequential(nn.Linear(self.fdim, scaled_dim), nn.ReLU())
        gate_weight_layer = nn.Bilinear(self.fdim, self.fdim, scaled_dim)
        out_layer = nn.Sequential(
            nn.Linear(scaled_dim, scaled_dim), nn.ReLU(), nn.Dropout(p=self.dropout)
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
        fdim=32,
        mmfdim=64,
        gate_graph_with_omic=1,
        gate_path=1,
        gate_graph=1,
        gate_omic=1,
        path_scale=1,
        graph_scale=1,
        omic_scale=1,
        dropout=0.25,
    ):
        # Register attributes
        super().__init__(device, fdim, dropout)
        self.gate_path = gate_path
        self.gate_graph = gate_graph
        self.gate_omic = gate_omic
        self.gate_graph_with_omic = gate_graph_with_omic

        path_scaled, graph_scaled, omic_scaled = (
            self.fdim // path_scale,
            self.fdim // graph_scale,
            self.fdim // omic_scale,
        )

        self.path_rescale, self.path_gate_weight, self.path_gated = (
            self._create_gate_layers(path_scaled)
        )
        self.graph_rescale, self.graph_gate_weight, self.graph_gated = (
            self._create_gate_layers(graph_scaled)
        )
        self.omic_rescale, self.omic_gate_weight, self.omic_gated = (
            self._create_gate_layers(omic_scaled)
        )

        self.post_fusion_dropout = nn.Dropout(p=self.dropout)
        self.encoder = nn.Sequential(
            nn.Linear(
                (path_scaled + 1) * (graph_scaled + 1) * (omic_scaled + 1), mmfdim
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(mmfdim, mmfdim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, **kwargs):
        f_path, f_graph, f_omic = kwargs["f_path"], kwargs["f_graph"], kwargs["f_omic"]
        p = self._rescale_and_gate(
            f_path,
            f_omic,
            self.path_rescale,
            self.path_gate_weight,
            self.path_gated,
            self.gate_path,
        )
        g = self._rescale_and_gate(
            f_graph,
            f_omic if self.gate_graph_with_omic else f_path,
            self.graph_rescale,
            self.graph_gate_weight,
            self.graph_gated,
            self.gate_graph,
        )
        o = self._rescale_and_gate(
            f_path,
            f_omic,
            self.omic_rescale,
            self.omic_gate_weight,
            self.omic_gated,
            self.gate_omic,
        )

        # Fusion
        p, g, o = self._append_one(p), self._append_one(g), self._append_one(o)
        fused = self._tensor_fusion([p, g, o])
        fused = self.post_fusion_dropout(fused)
        fused = self.encoder(fused)
        return fused


class Bimodal(TensorFusion):
    def __init__(
        self,
        device,
        fdim=32,
        mmfdim=64,
        gate1=1,
        gate2=1,
        scale_dim1=1,
        scale_dim2=1,
        dropout=0.25,
    ):
        super().__init__(device, fdim, dropout)
        self.gate1 = gate1
        self.gate2 = gate2
        self.device = device

        dim1_scaled, dim2_scaled = self.fdim // scale_dim1, self.fdim // scale_dim2

        self.rescale_1, self.gate_weight_1, self.out_1 = self._create_gate_layers(
            dim1_scaled
        )
        self.rescale_2, self.gate_weight_2, self.out_2 = self._create_gate_layers(
            dim2_scaled
        )

        self.post_fusion_dropout = nn.Dropout(p=self.dropout)
        self.encoder = nn.Sequential(
            nn.Linear((dim1_scaled + 1) * (dim2_scaled + 1), mmfdim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(mmfdim, mmfdim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, **kwargs):
        vec1 = kwargs["f_path"] or kwargs["f_graph"]
        # Omic is always present and takes the second position
        vec2 = kwargs["f_omic"]
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
        fused = self.post_fusion_dropout(fused)
        fused = self.encoder(fused)
        return fused
