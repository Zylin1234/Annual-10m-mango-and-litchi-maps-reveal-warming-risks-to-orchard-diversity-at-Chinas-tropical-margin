from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return next(g for g in (16, 8, 4, 2, 1) if channels % g == 0)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, groups=groups, bias=False)
        self.norm = nn.GroupNorm(_groups(out_ch), out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDW(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(channels, channels, 3, groups=channels),
            ConvGNAct(channels, channels, 1),
            nn.Dropout2d(dropout),
        )
        self.gamma = nn.Parameter(torch.tensor([0.25]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.body(x)


class PixelRMSNorm(nn.Module):
    """Normalize only the adapter residual; the identity AEF path is untouched."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-6)


class IdentifiableAEFAdapter(nn.Module):
    """Identity-initialized low-rank adaptation on the 64 raw AEF dimensions.

    The former class-gate multiplied by a free classifier weight, making both
    factors non-identifiable.  This adapter removes that factorization and does
    not claim that a learned scale is a causal channel importance score.
    """

    def __init__(self, channels: int = 64, rank: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm = PixelRMSNorm()
        self.down = nn.Conv2d(channels, rank, 1, bias=True)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        self.channel_scale = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.tensor(0.10))
        self.dropout = nn.Dropout2d(dropout)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bounded_scale = 1.0 + 0.10 * torch.tanh(self.channel_scale).view(1, -1, 1, 1)
        delta = self.up(F.gelu(self.down(self.norm(x))))
        y = x * bounded_scale + self.gamma * self.dropout(delta)
        return y, {"adapter_scale": bounded_scale, "adapter_delta_rms": delta.square().mean().sqrt()}


class DualPathOCE(nn.Module):
    """Resolution-preserving orchard context at 30, 70 and 150 m support.

    Context is strong for the orchard/background head and deliberately weak for
    the species head, preventing orchard smoothing from erasing AEF species cues.
    No patch-global branch is used, so predictions do not depend on arbitrary
    128x128 crop composition.
    """

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.local = nn.Sequential(ConvGNAct(channels, channels, 3, groups=channels), ConvGNAct(channels, channels, 1))
        self.medium = ConvGNAct(channels, channels, 1)
        self.broad = ConvGNAct(channels, channels, 1)
        self.gate = nn.Conv2d(channels, 3, 1, bias=True)
        self.fuse = nn.Sequential(ConvGNAct(channels, channels, 1), nn.Dropout2d(dropout))
        self.gamma = nn.Parameter(torch.tensor(0.10))
        self.species_context_logit = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid=0.10
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def _mean(x: torch.Tensor, kernel: int) -> torch.Tensor:
        return F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        branches = (self.local(x), self.medium(self._mean(x, 7)), self.broad(self._mean(x, 15)))
        weights = torch.softmax(self.gate(x), dim=1)
        mixed = sum(branch * weights[:, i : i + 1] for i, branch in enumerate(branches))
        context_delta = self.gamma * self.fuse(mixed)
        orchard_features = x + context_delta
        species_scale = torch.sigmoid(self.species_context_logit)
        species_features = x + species_scale * context_delta
        return orchard_features, species_features, {
            "context_weights": weights,
            "species_context_scale": species_scale,
            "context_delta_rms": context_delta.square().mean().sqrt(),
        }


class SegHead(nn.Sequential):
    def __init__(self, channels: int, classes: int, dropout: float):
        super().__init__(ConvGNAct(channels, channels, 3), nn.Dropout2d(dropout), nn.Conv2d(channels, classes, 1))


@dataclass(frozen=True)
class Variant:
    adapter: bool
    oce: bool
    hierarchical: bool


VARIANTS = {
    "C0_baseline": Variant(False, False, False),
    "C1_adapter": Variant(True, False, False),
    "C2_adapter_oce": Variant(True, True, False),
    "C3_hierarchical": Variant(True, True, True),
    "C3_minus_adapter": Variant(False, True, True),
    "C3_minus_oce": Variant(True, False, True),
}


def compose_hierarchical_logits(orchard_logit: torch.Tensor, species_logits: torch.Tensor) -> torch.Tensor:
    """Return normalized log P(background, mango, lychee).

    P(bg)=1-q, P(mango)=q*r_m, P(lychee)=q*r_l, where q is orchard
    probability and r is the species probability conditional on orchard.
    """

    log_bg = F.logsigmoid(-orchard_logit)
    log_orchard = F.logsigmoid(orchard_logit)
    log_species = F.log_softmax(species_logits, dim=1)
    return torch.cat((log_bg, log_orchard + log_species), dim=1)


class RefinedAEFOCENet(nn.Module):
    def __init__(self, cfg, variant: str | None = None):
        super().__init__()
        self.variant_name = variant or cfg.variant
        self.variant = VARIANTS[self.variant_name]
        self.adapter = IdentifiableAEFAdapter(64, cfg.adapter_rank, cfg.dropout) if self.variant.adapter else None
        self.stem = nn.Sequential(ConvGNAct(65, cfg.base_dim, 1), ConvGNAct(cfg.base_dim, cfg.base_dim, 3))
        self.blocks = nn.Sequential(*[ResidualDW(cfg.base_dim, cfg.dropout) for _ in range(cfg.num_blocks)])
        self.oce = DualPathOCE(cfg.base_dim, cfg.dropout) if self.variant.oce else None
        if self.variant.hierarchical:
            self.orchard_head = SegHead(cfg.base_dim, 1, cfg.dropout)
            self.species_head = SegHead(cfg.base_dim, 2, cfg.dropout)
            self.direct_head = None
        else:
            self.direct_head = SegHead(cfg.base_dim, 3, cfg.dropout)
            self.orchard_head = self.species_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 65:
            raise ValueError(f"Expected [B,65,H,W], got {tuple(x.shape)}")
        raw, valid = x[:, :64], x[:, 64:65].clamp(0, 1)
        raw = raw * valid
        out: dict[str, torch.Tensor] = {}
        if self.adapter is not None:
            raw, diagnostics = self.adapter(raw)
            out.update(diagnostics)
        semantic = self.blocks(self.stem(torch.cat((raw, valid), dim=1)))
        orchard_features = species_features = semantic
        if self.oce is not None:
            orchard_features, species_features, diagnostics = self.oce(semantic)
            out.update(diagnostics)
        out["semantic_features"] = semantic
        if self.variant.hierarchical:
            orchard_logit = self.orchard_head(orchard_features)
            species_logits = self.species_head(species_features)
            out.update(
                orchard_logit=orchard_logit,
                species_logits=species_logits,
                logits=compose_hierarchical_logits(orchard_logit, species_logits),
            )
        else:
            out["logits"] = self.direct_head(orchard_features)
        return out


def build_model(cfg, variant: str | None = None) -> RefinedAEFOCENet:
    return RefinedAEFOCENet(cfg, variant)
