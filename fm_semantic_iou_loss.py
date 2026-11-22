import torch
import torch.nn as nn

from mmcv.ops import diff_iou_rotated_2d
import torch.distributed as dist
from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class FMSemanticIoULoss(nn.Module):
    """Fractional-Modulated Semantic IoU Loss (FMSemanticIoULoss).

    This loss fuses:

    1) Semantic-geometric coupling from RoI features:

        s1 = (cos(sem_pred, sem_gt) + 1) / 2  in [0, 1]
        s2 = (cos(sem_pred_ctx, sem_gt_ctx) + 1) / 2  (optional)
        s  = (1 - lambda_ctx) * s1 + lambda_ctx * s2

        u  = IoU_rotated(pred_bboxes, target_bboxes)

        L_base = (1 - s) * (1 - u)

    2) Fractional-order modulation on the semantic IoU sequence:

        sIoU = s + (1 - s) * u
        D_t^nu(sIoU) ≈ sum_{k=0}^M c_k * sIoU_{t-k}
        W_t = exp(-lambda_ * D_t^nu)

    Final loss:

        L = W_t * L_base

    where W_t is a batch-level scalar (detached) computed from the history
    of mean sIoU over recent batches.

    Args:
        nu (float): Fractional order (0 < nu < 1).
        lambda_ (float): Strength of fractional modulation.
        lambda_ctx (float): Weight for contextual semantic similarity.
        M (int): History length for batch-level sIoU means.
        eps (float): Small value for numerical stability.
        reduction (str): 'none', 'mean' or 'sum'.
        loss_weight (float): Overall loss weight.
        use_fractional (bool): If False, disables fractional modulation
            and uses only L_base.
    """

    def __init__(self,
                 nu: float = 0.8,
                 lambda_: float = 1.0,
                 lambda_ctx: float = 0.5,
                 M: int = 3,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 use_fractional: bool = True,
                 alpha: float = 0.1,
                 min_sem_weight: float = 0.05):
        super().__init__()
        assert 0.0 < nu < 1.0, 'Fractional order nu must be in (0, 1).'
        assert M >= 1, 'History length M must be >= 1.'
        assert reduction in ('none', 'mean', 'sum')

        self.nu = float(nu)
        self.lambda_ = float(lambda_)
        self.lambda_ctx = float(lambda_ctx)
        self.M = int(M)
        self.eps = float(eps)
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self.use_fractional = use_fractional
        self._debug_step = 0
        # Scale for intra-region semantic distance in s1 = exp(-alpha * d)
        self.alpha = float(alpha)
        # Lower bound for (1 - s) so that semantic gating does not
        # completely kill geometric gradients.
        self.min_sem_weight = float(min_sem_weight)

        # Pre-compute Grünwald–Letnikov coefficients c_0 ... c_M
        coeffs = [1.0]
        for k in range(1, self.M + 1):
            coeffs.append(coeffs[-1] * (k - 1 - self.nu) / k)

        # History buffer: store mean sIoU of last M batches
        self.register_buffer('siou_history',
                             torch.full((self.M,), 0.5, dtype=torch.float32))
        self.register_buffer('gl_coeffs',
                             torch.tensor(coeffs, dtype=torch.float32))

    def _fractional_trend(self, siou_mean: torch.Tensor) -> torch.Tensor:
        """Compute fractional-order trend D_t and update sIoU history."""
        D_t = self.gl_coeffs[0] * siou_mean
        if self.M > 0:
            D_t = D_t + torch.dot(self.gl_coeffs[1:], self.siou_history)

        # Roll history and insert current mean at position 0
        self.siou_history = torch.roll(self.siou_history, shifts=1, dims=0)
        self.siou_history[0] = siou_mean
        return D_t

    def forward(self,
                pred_bboxes: torch.Tensor,
                target_bboxes: torch.Tensor,
                sem_pred: torch.Tensor,
                sem_gt: torch.Tensor,
                sem_pred_ctx: torch.Tensor = None,
                sem_gt_ctx: torch.Tensor = None,
                weight: torch.Tensor = None,
                avg_factor=None,
                reduction_override: str = None):
        """Forward function.

        Args:
            pred_bboxes (Tensor): Predicted rbboxes, shape (N, 5).
            target_bboxes (Tensor): GT rbboxes, shape (N, 5).
            sem_pred (Tensor): Semantic embeddings for predicted RoIs,
                shape (N, C), L2 normalized.
            sem_gt (Tensor): Semantic embeddings for GT RoIs, shape (N, C).
            sem_pred_ctx (Tensor, optional): Contextual semantic embeddings
                for predicted RoIs, shape (N, C) or (N, C, H, W). Defaults
                to None.
            sem_gt_ctx (Tensor, optional): Contextual semantic embeddings
                for GT RoIs, shape (N, C) or (N, C, H, W). Defaults to None.
            weight (Tensor, optional): Per-sample weights, shape (N,) or
                (N, 5). Defaults to None.
            avg_factor (int | float, optional): Average factor used when
                reduction is 'mean'. Defaults to None.
            reduction_override (str, optional): Override reduction method.
        """
        assert pred_bboxes.size(-1) == 5
        assert target_bboxes.size(-1) == 5
        assert sem_pred.shape == sem_gt.shape

        reduction = reduction_override if reduction_override else self.reduction

        # Enforce ∂s/∂θ ≈ 0 for the detection parameters by preventing
        # gradients from flowing through semantic descriptors.
        sem_pred = sem_pred.detach()
        sem_gt = sem_gt.detach()
        if sem_pred_ctx is not None and sem_gt_ctx is not None:
            sem_pred_ctx = sem_pred_ctx.detach()
            sem_gt_ctx = sem_gt_ctx.detach()

        # 1) Rotated IoU u in [0, 1]
        ious = diff_iou_rotated_2d(
            pred_bboxes.unsqueeze(0), target_bboxes.unsqueeze(0))
        ious = ious.squeeze(0).clamp(self.eps, 1.0 - self.eps)

        # 2) Fine-grained semantic similarity s1 via intra-region
        # mean squared feature distance on raw RoI features.
        # Use mean over channels to keep scale stable across C.
        dist_sq = (sem_pred - sem_gt).pow(2).mean(dim=-1)
        s1 = torch.exp(-self.alpha * dist_sq)  # in (0, 1]

        # 3) Optional contextual semantic similarity s2 via cosine
        # similarity on context features.
        if sem_pred_ctx is not None and sem_gt_ctx is not None:
            assert sem_pred_ctx.shape == sem_gt_ctx.shape
            if sem_pred_ctx.dim() == 2:
                # Global cosine on flattened descriptors: (N, C)
                dot = (sem_pred_ctx * sem_gt_ctx).sum(dim=-1)
                norm_p = sem_pred_ctx.norm(dim=-1).clamp_min(self.eps)
                norm_g = sem_gt_ctx.norm(dim=-1).clamp_min(self.eps)
                cos_sim_ctx = dot / (norm_p * norm_g)
                s2 = (cos_sim_ctx + 1.0) * 0.5
            elif sem_pred_ctx.dim() == 4:
                # Per-location cosine followed by averaging:
                # sem_pred_ctx, sem_gt_ctx: (N, C, H, W)
                N_ctx, C_ctx, H_ctx, W_ctx = sem_pred_ctx.shape
                f_p = sem_pred_ctx.view(N_ctx, C_ctx, -1)  # (N, C, P)
                f_g = sem_gt_ctx.view(N_ctx, C_ctx, -1)    # (N, C, P)
                # Per-location cosine along channel dimension
                numer = (f_p * f_g).sum(dim=1)             # (N, P)
                norm_p = f_p.norm(dim=1).clamp_min(self.eps)  # (N, P)
                norm_g = f_g.norm(dim=1).clamp_min(self.eps)  # (N, P)
                cos_loc = numer / (norm_p * norm_g)        # (N, P)
                cos_sim_ctx = cos_loc.mean(dim=1)          # (N,)
                s2 = (cos_sim_ctx + 1.0) * 0.5
            else:
                raise ValueError(
                    f'Unsupported context embedding dim: {sem_pred_ctx.dim()}')
            s = (1.0 - self.lambda_ctx) * s1 + self.lambda_ctx * s2
        else:
            s = s1

        s = s.clamp(0.0, 1.0)

        # 4) Semantic IoU and base loss. We view (1 - s) mainly as a
        # semantic weight on geometric error, and clip it from below so
        # that the geometric gradient cannot vanish completely even when
        # s ≈ 1.
        sIoU = s + (1.0 - s) * ious
        sem_weight = (1.0 - s).clamp_min(self.min_sem_weight)
        base_loss = sem_weight * (1.0 - ious)
        
        


        self._debug_step += 1
        if self.training and self._debug_step % 50 == 0:
            # 再做上面的 rank 判断和打印

            rank = 0
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            if rank == 0:
                s_mean = s.mean().item()
                s_min = s.min().item()
                s_max = s.max().item()
                u_mean = ious.mean().item()
                u_min = ious.min().item()
                u_max = ious.max().item()
                print(
                    f'[FMSemIoU debug] s_mean={s_mean:.3f}, s_min={s_min:.3f}, s_max={s_max:.3f}, '
                    f'u_mean={u_mean:.3f}, u_min={u_min:.3f}, u_max={u_max:.3f}'
                )


        # Prepare effective sample weights
        eff_weight = None
        if weight is not None:
            if weight.dim() > 1:
                eff_weight = weight.mean(-1)
            else:
                eff_weight = weight

        # 5) Fractional-order modulation based on sIoU trend
        if self.use_fractional:
            with torch.no_grad():
                if eff_weight is not None:
                    pos_mask = eff_weight > 0
                    if torch.any(pos_mask):
                        siou_mean = sIoU[pos_mask].mean()
                    else:
                        siou_mean = sIoU.new_tensor(0.5)
                else:
                    siou_mean = sIoU.mean()
                D_t = self._fractional_trend(siou_mean)
            W_t = torch.exp(-self.lambda_ * D_t).detach()
        else:
            W_t = 1.0

        loss = W_t * base_loss

        if eff_weight is not None:
            loss = loss * eff_weight

        if reduction == 'mean':
            loss = loss.mean() if avg_factor is None else loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss
