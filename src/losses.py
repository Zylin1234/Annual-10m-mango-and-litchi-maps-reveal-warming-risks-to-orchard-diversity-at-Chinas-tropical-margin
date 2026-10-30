from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha, gamma: float, ignore_index: int) -> torch.Tensor:
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0
    safe = target.masked_fill(~valid, 0)
    log_prob = F.log_softmax(logits, dim=1)
    log_pt = log_prob.gather(1, safe.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    weights = torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype)[safe]
    return (-weights * (1 - pt).pow(gamma) * log_pt)[valid].mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, classes: int, alpha: float, beta: float, ignore_index: int) -> torch.Tensor:
    valid = target != ignore_index
    safe = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe, classes).permute(0, 3, 1, 2).to(logits.dtype) * valid.unsqueeze(1)
    probability = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
    dims = (0, 2, 3)
    tp = (probability * one_hot).sum(dims)
    fp = (probability * (1 - one_hot)).sum(dims)
    fn = ((1 - probability) * one_hot).sum(dims)
    return 1 - ((tp + 1) / (tp + alpha * fp + beta * fn + 1)).mean()


def _ramp(epoch: int, start: int, duration: int) -> float:
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start + 1) / max(duration, 1))


class HierarchicalSegmentationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _species_margin(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        foreground = (target == 1) | (target == 2)
        if not foreground.any():
            return logits.sum() * 0
        desired = torch.where(target == 1, logits[:, 0] - logits[:, 1], logits[:, 1] - logits[:, 0])
        values = F.softplus(self.cfg.species_margin - desired[foreground])
        k = max(1, math.ceil(values.numel() * self.cfg.margin_topk_ratio))
        return values.topk(k, largest=True).values.mean()

    def forward(self, outputs: dict[str, torch.Tensor], target: torch.Tensor, epoch: int = 0):
        logits = outputs["logits"]
        main_focal = focal_loss(logits, target, self.cfg.focal_alpha, self.cfg.focal_gamma, self.cfg.ignore_index)
        main_tversky = tversky_loss(
            logits, target, self.cfg.num_classes, self.cfg.tversky_alpha, self.cfg.tversky_beta, self.cfg.ignore_index
        )
        loss = main_focal + self.cfg.lambda_tversky * main_tversky
        logs = {"main_focal": main_focal.detach(), "main_tversky": main_tversky.detach()}

        if "orchard_logit" in outputs:
            valid = target != self.cfg.ignore_index
            orchard_target = (target > 0).to(logits.dtype)
            orchard_raw = F.binary_cross_entropy_with_logits(outputs["orchard_logit"].squeeze(1), orchard_target, reduction="none")
            orchard_loss = orchard_raw[valid].mean() if valid.any() else orchard_raw.sum() * 0
            foreground = valid & (target > 0)
            if foreground.any():
                species_target = target[foreground] - 1
                species_pixels = outputs["species_logits"].permute(0, 2, 3, 1)[foreground]
                species_weight = torch.tensor((1.0, 1.10), device=logits.device, dtype=logits.dtype)
                species_loss = F.cross_entropy(species_pixels, species_target, weight=species_weight)
                margin_loss = self._species_margin(outputs["species_logits"], target)
            else:
                species_loss = outputs["species_logits"].sum() * 0
                margin_loss = species_loss
            margin_weight = self.cfg.lambda_species_margin * _ramp(
                epoch, self.cfg.margin_start_epoch, self.cfg.margin_ramp_epochs
            )
            loss = loss + self.cfg.lambda_orchard_aux * orchard_loss + self.cfg.lambda_species_aux * species_loss + margin_weight * margin_loss
            logs.update(
                orchard_aux=orchard_loss.detach(),
                species_aux=species_loss.detach(),
                species_margin=margin_loss.detach(),
                species_margin_weight=torch.tensor(margin_weight, device=logits.device),
            )
        logs["total"] = loss.detach()
        return loss, logs

