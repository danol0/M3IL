from argparse import Namespace

import torch
import torch.nn as nn

"""
This module reimplements the Kroneker fusion described in the paper. Hyperparameters have been hardcoded
to match the paper's implementation as they have not been tuned for this project (see section 1.2 of report).
Reference: https://github.com/mahmoodlab/PathomicFusion
"""


def define_tensor_fusion(opt: Namespace, mmfdim: int) -> nn.Module:
    """Reproduces the Kroneker fusion."""

    gate_path = (
        False
        if opt.task == "grad" and opt.model in ("pathomic", "pathgraphomic")
        else True
    )
    gate_graph = False if opt.task == "grad" and opt.model == "graphomic" else True
    gate_omic = (
        False if opt.task == "surv" and opt.model in ("pathomic", "graphomic") else True
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
            gate_graph_with_omic=True if opt.task == "surv" else False,
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
    def __init__(self, device: torch.device, fdim: int, dropout: float) -> None:
        """Implements shared operations for bimodal and trimodal tensor fusion."""

        super().__init__()
        self.device = device
        self.fdim = fdim
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()

    def _tensor_fusion(self, tensors: list) -> torch.Tensor:
        """Kronecker product fusion of a list of encoded modalities."""

        if tensors[0].dim() == 2:
            if len(tensors) == 2:
                f = torch.einsum("bi,bj->bij", *tensors).flatten(start_dim=1)
            elif len(tensors) == 3:
                f = torch.einsum("bi,bj,bk->bijk", *tensors).flatten(start_dim=1)
            else:
                raise NotImplementedError("Only bimodal and trimodal fusion supported.")

        elif tensors[0].dim() == 3:
            if len(tensors) == 2:
                f = torch.einsum("bsi,bsj->bsij", *tensors).flatten(start_dim=2)
            elif len(tensors) == 3:
                f = torch.einsum("bsi,bsj,bsk->bsijk", *tensors).flatten(start_dim=2)
            else:
                raise NotImplementedError("Only bimodal and trimodal fusion supported.")
        else:
            raise NotImplementedError("Only 2D and 3D tensors supported.")

        return f

    def _create_gate_layers(self, scaled_dim: int) -> tuple:
        """Creates layers required for feature gating."""

        rescale_layer = nn.Sequential(nn.Linear(self.fdim, scaled_dim), nn.ReLU())
        gate_weight_layer = nn.Bilinear(self.fdim, self.fdim, scaled_dim)
        out_layer = nn.Sequential(
            nn.Linear(scaled_dim, scaled_dim), nn.ReLU(), nn.Dropout(p=self.dropout)
        )
        return rescale_layer, gate_weight_layer, out_layer

    def _rescale_and_gate(
        self,
        x: torch.Tensor,
        x_gate: torch.Tensor,
        rescale_layer: nn.Module,
        gate_layer: nn.Module,
        out_layer: nn.Module,
        gate: int,
    ) -> torch.Tensor:
        """
        Rescales and gates a modality.

        NOTE: Gating behaviour has been changed from the paper.
        Specifically, the paper does not apply the rescaling layer if the gate is off.
        This would result in an error if a mode is rescaled but not gated, as the out layer
        expects a rescaled input. This implementation applies the rescaling layer regardless
        of the gate state, meaning that ungated modalities undergo an additional transformation
        vs the paper's implementation.
        """
        o = rescale_layer(x)  # (*, fdim) -> (*, scaled_dim)
        w = (
            self.sigmoid(gate_layer(x, x_gate)) if gate else 1
        )  # (*, fdim) -> (*, scaled_dim)
        o = out_layer(w * o)  # (*, scaled_dim) -> (*, scaled_dim)
        return o

    def _append_one(self, x: torch.Tensor) -> torch.Tensor:
        _one = torch.ones((*x.shape[:-1], 1), device=self.device)
        return torch.cat((x, _one), dim=-1)


class Trimodal(TensorFusion):
    def __init__(
        self,
        device: torch.device,
        fdim: int,
        mmfdim: int,
        gate_graph_with_omic: bool,
        gate_path: bool,
        gate_graph: bool,
        gate_omic: bool,
        path_scale: int,
        graph_scale: int,
        omic_scale: int,
        dropout: float,
    ) -> None:
        """
        Trimodal tensor fusion.

        Args:
            device (torch.device): Pytorch backend.
            fdim (int): Dimension of input feature vector for each modality.
            mmfdim (int): Dimension of the output multimodal feature vector.
            gate_graph_with_omic (bool): True gates graph with omic, False gates graph with path.
            gate_path (bool): Whether to gate the path feature vector.
            gate_graph (bool): Whether to gate the graph feature vector.
            gate_omic (bool): Whether to gate the omic feature vector.
            path_scale (int): Scaling factor for the path feature vector.
            graph_scale (int): Scaling factor for the graph feature vector.
            omic_scale (int): Scaling factor for the omic feature vector.
            dropout (float): Dropout rate.
        """
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

    def forward(self, **kwargs: dict) -> torch.Tensor:
        f_path, f_graph, f_omic = kwargs["f_path"], kwargs["f_graph"], kwargs["f_omic"]

        # if f_path.dim() == 3 and f_omic.dim() == 2:
        #     f_omic = f_omic.unsqueeze(1)

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
        device: torch.device,
        fdim: int,
        mmfdim: int,
        gate1: bool,
        gate2: bool,
        scale_dim1: int,
        scale_dim2: int,
        dropout: float,
    ) -> None:
        """
        Bimodal tensor fusion.

        Args:
            device (torch.device): Pytorch backend.
            fdim (int): Dimension of input feature vector for each modality.
            mmfdim (int): Dimension of the output multimodal feature vector.
            gate1 (bool): Whether to gate the first feature vector.
            gate2 (bool): Whether to gate the second feature vector.
            scale_dim1 (int): Scaling factor for the first feature vector.
            scale_dim2 (int): Scaling factor for the second feature vector.
            dropout (float): Dropout rate.
        """
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

    def forward(self, **kwargs: dict) -> torch.Tensor:
        vec1 = kwargs["f_path"] if kwargs["f_path"] is not None else kwargs["f_graph"]
        # Omic is always present and takes the second position
        f_omic = kwargs["f_omic"]

        if vec1.dim() == 3 and f_omic.dim() == 2:
            f_omic = f_omic.unsqueeze(1)

        o1 = self._rescale_and_gate(
            vec1,
            f_omic,
            self.rescale_1,
            self.gate_weight_1,
            self.out_1,
            self.gate1,
        )

        o2 = self._rescale_and_gate(
            f_omic,
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
