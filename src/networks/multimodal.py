from argparse import Namespace

import torch
from torch import nn

from src.networks.tensor_fusion import define_tensor_fusion
from src.networks.unimodal import (
    FFN,
    GNN,
    BaseEncoder,
    MaskedAttentionPool,
    MaskedMeanPool,
    GraphAttentionPool,
)

def print_load(ckpt: str, device: torch.device) -> dict:
    print(f"Loading {ckpt}")
    try:
        return torch.load(ckpt, map_location=device)
    except FileNotFoundError:
        print(f"Checkpoint {ckpt} not found. Exiting...")
        exit(1)


class FlexibleFusion(BaseEncoder):
    def __init__(
        self, opt: Namespace, fdim: int = 32, mmfdim: int = 64, local: str = None
    ) -> None:
        """
        Dynamic multimodal fusion of path, graph and omics data:
        1. Encodes each modality with (pre-trained) modality-specific encoders
        2. Aggregates path images if necessary
        3. Calls a fusion architecture to combine the multimodal features

        Args:
            opt (Namespace): Command line arguments, specifying model and task
            fdim (int): Dimension of feature vector for each modality.
            mmfdim (int): Dimension of fused multimodal feature vector.
        """
        super().__init__(mmfdim, local)
        self.device = opt.device

        # --- Feature encoders ---
        omic_ckpt, graph_ckpt = self.get_unimodal_ckpts(opt)

        if "omic" in opt.model:
            self.omic_net = FFN(
                xdim=320 if opt.rna else 80, fdim=fdim, dropout=opt.dropout
            )
            self.omic_net.load_state_dict(omic_ckpt["model"])
            self.omic_net = self.omic_net.to(opt.device)
            self.omic_net.freeze(True)

        if "graph" in opt.model:
            # Pooling is handled within the GNN model
            self.graph_net = GNN(fdim=fdim, pool=opt.graph_pool, dropout=opt.dropout)
            self.graph_net.load_state_dict(graph_ckpt["model"])
            self.graph_net.freeze(True)
            if opt.graph_pool == "attn":
                # Reset attention weights
                self.graph_net.aggregate = GraphAttentionPool(fdim=fdim, hdim=fdim)
            self.graph_net = self.graph_net.to(opt.device)

        if "path" in opt.model:
            assert (
                opt.pre_encoded_path
            ), "Pre-extracted path features must be used for MM models."
            # Unlike graphs, path images are not aggregated in their encoder/as pre-extracted features
            self.aggregate = nn.Identity()
            if opt.mil == "global":
                self.aggregate = (
                    MaskedAttentionPool(fdim=fdim, hdim=fdim // 2, dropout=opt.dropout)
                    if opt.pool == "attn"
                    else MaskedMeanPool()
                )

        # --- Fusion ---
        self.fusion = define_tensor_fusion(opt, mmfdim=mmfdim)

    def forward(self, **kwargs: dict) -> torch.Tensor:
        p, g, o = None, None, None

        if hasattr(self, "omic_net"):
            o = self.omic_net.get_latents(**kwargs)

        if hasattr(self, "graph_net"):
            g = self.graph_net.get_latents(**kwargs)

        if hasattr(self, "aggregate"):
            p = self.aggregate(kwargs["x_path"])

        p, g, o = self.align_MM_local_MIL(p, g, o) if self.local else (p, g, o)
        x = self.fusion(f_omic=o, f_graph=g, f_path=p)
        return self.predict(x, mask=self.get_mask(p, g))

    def get_mask(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Generates mask for 0 padded multimodal inputs, which is only used in local MIL."""

        mask = None
        if self.local:
            # When both path and graph are present, the masks will align with path
            if p is not None:
                mask = torch.any(p != 0, dim=-1)
            elif g is not None:
                mask = torch.any(g != 0, dim=-1)
        return mask

    def align_MM_local_MIL(
        self, p: torch.Tensor, g: torch.Tensor, o: torch.Tensor
    ) -> tuple:
        """Aligns samples across modalities as done in original architecture for local MIL,
        such that they are all shape (batch, samples, features) with corresponding instances
        in each index."""
        max_sample = max(
            p.size(1) if p is not None else 0, g.size(1) if g is not None else 0
        )
        if o is not None:
            o = o.unsqueeze(1)
            o = o.repeat(1, max_sample, 1)
        if g is not None and g.size(1) != max_sample:
            assert g.size(1) * 9 == p.size(1)
            g = g.repeat_interleave(9, dim=1, output_size=max_sample)
        return p, g, o

    def l1(self) -> torch.Tensor:
        # We only regularise the omic network in MM models
        return sum(torch.abs(W).sum() for W in self.omic_net.parameters())

    @staticmethod
    def get_unimodal_ckpts(opt: Namespace) -> tuple:
        rna = "_rna" if opt.rna else ""
        omic_ckpt = (
            print_load(
                f"{opt.save_dir}/{opt.task}/omic{rna}/omic_{opt.k}.pt",
                device=opt.device,
            )
            if "omic" in opt.model
            else None
        )
        pool = f"_{opt.pool}" if opt.pool != "mean" else ""
        graph_ckpt = (
            print_load(
                f"{opt.save_dir}/{opt.task}/graph_{opt.mil}{pool}/graph_{opt.k}.pt",
                device=opt.device,
            )
            if "graph" in opt.model
            else None
        )
        return omic_ckpt, graph_ckpt
