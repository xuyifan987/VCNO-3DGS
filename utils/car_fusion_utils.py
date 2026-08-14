import torch
import torch.nn.functional as F


def robust_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Map a tensor to [0, 1] with percentile clipping for outlier robustness."""
    x = x.float()
    if x.numel() == 0:
        return x
    flat = x.reshape(-1)
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.95)
    x = (x - lo) / (hi - lo + eps)
    return x.clamp_(0.0, 1.0)


def local_mean(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    original_dim = x.dim()
    if original_dim == 2:
        x = x[None, None]
    elif original_dim == 3:
        x = x[None]
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.avg_pool2d(x, kernel_size=kernel_size, stride=1)
    if original_dim == 2:
        return y[0, 0]
    if original_dim == 3:
        return y[0]
    return y


@torch.no_grad()
def image_reliability_map(render_image: torch.Tensor,
                          gt_image: torch.Tensor,
                          alpha: torch.Tensor = None,
                          kernel_size: int = 9,
                          residual_temperature: float = 0.20,
                          detail_preserve: float = 0.25,
                          edge_boost: float = 0.50) -> torch.Tensor:
    """
    Estimate whether a pixel is useful for densification.

    High residual pixels are not always reliable: reflective highlights, moving
    background, poor pose regions, and empty alpha areas can trigger false growth.
    This map keeps densification focused on pixels that are visible and locally
    consistent, while preserving GT edge regions that are usually legitimate car
    details rather than noise.
    """
    residual = torch.mean(torch.abs(render_image.float() - gt_image.float()), dim=0)
    residual = local_mean(residual, kernel_size)
    residual_score = torch.exp(-residual / max(residual_temperature, 1e-6))

    if alpha is None:
        alpha_score = torch.ones_like(residual_score)
    else:
        if alpha.dim() == 3:
            alpha = alpha[0]
        alpha_score = robust_normalize(local_mean(alpha.float(), kernel_size))

    reliability = torch.sqrt((residual_score * alpha_score).clamp_min(0.0))
    detail_preserve = min(max(float(detail_preserve), 0.0), 1.0)
    edge_boost = max(float(edge_boost), 0.0)
    if detail_preserve > 0.0 and edge_boost > 0.0:
        gt_lum = _luminance(gt_image)[0]
        edge_x, edge_y = _sobel_xy(gt_lum[None])
        edge = torch.sqrt(edge_x[0, 0].pow(2) + edge_y[0, 0].pow(2) + 1e-6)
        edge_score = robust_normalize(local_mean(edge, kernel_size))
        detail_score = (edge_score * alpha_score).clamp(0.0, 1.0)
        reliability = torch.maximum(reliability, detail_preserve * edge_boost * detail_score)
    return reliability.clamp_(0.0, 1.0)


@torch.no_grad()
def fuse_frequency_reliability_error(frequency_error: torch.Tensor,
                                     reliability: torch.Tensor,
                                     error_weight: float = 1.0,
                                     reliability_power: float = 1.5,
                                     reliability_floor: float = 0.0,
                                     adaptive_floor_strength: float = 0.0,
                                     adaptive_floor_power: float = 1.0,
                                     adaptive_floor_detail_power: float = 0.0) -> torch.Tensor:
    """Suppress high-frequency error in unreliable pixels before inverse projection.

    reliability_floor keeps a residual raw-error route alive for dense object captures
    where highlights and thin contours can be reliable high-frequency detail.
    """
    frequency_error = robust_normalize(frequency_error.float())
    reliability = reliability.float().clamp(0.0, 1.0)
    reliability_floor = min(max(float(reliability_floor), 0.0), 1.0)
    reliability_gate = reliability.pow(reliability_power)
    if reliability_floor > 0.0:
        reliability_gate = reliability_floor + (1.0 - reliability_floor) * reliability_gate
    adaptive_floor_strength = min(max(float(adaptive_floor_strength), 0.0), 1.0)
    if adaptive_floor_strength > 0.0:
        adaptive_floor_power = max(float(adaptive_floor_power), 0.0)
        adaptive_floor_detail_power = max(float(adaptive_floor_detail_power), 0.0)
        residual_route = (1.0 - reliability).pow(adaptive_floor_power)
        if adaptive_floor_detail_power > 0.0:
            residual_route = residual_route * frequency_error.pow(adaptive_floor_detail_power)
        reliability_gate = torch.maximum(reliability_gate, adaptive_floor_strength * residual_route)
    return error_weight * frequency_error * reliability_gate


def _as_nchw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image[None]
    return image


def _luminance(image: torch.Tensor) -> torch.Tensor:
    image = _as_nchw(image).float()
    if image.shape[1] < 3:
        return image[:, :1]
    coeffs = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    return (image[:, :3] * coeffs).sum(dim=1, keepdim=True)


def _sobel_xy(x: torch.Tensor):
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)


def reliability_weighted_structure_loss(render_image: torch.Tensor,
                                        gt_image: torch.Tensor,
                                        reliability: torch.Tensor = None,
                                        scales=(1, 2, 4),
                                        reliability_power: float = 1.0,
                                        min_reliability: float = 0.0,
                                        edge_boost: float = 1.0,
                                        eps: float = 1e-3) -> torch.Tensor:
    """
    Multi-scale edge/structure consistency for reliable vehicle regions.

    The loss compares Sobel gradients rather than raw colors, so it complements
    the photometric L1/SSIM objective by preserving decals, glass boundaries,
    wheel rims, and body edges without forcing noisy residual pixels.
    """
    render_lum = _luminance(render_image)
    gt_lum = _luminance(gt_image)
    if reliability is None:
        rel = torch.ones_like(render_lum[:, :1])
    else:
        rel = reliability.float()
        if rel.dim() == 2:
            rel = rel[None, None]
        elif rel.dim() == 3:
            rel = rel[None] if rel.shape[0] != 1 else rel[:, None]
        rel = rel.to(render_lum.device).clamp(0.0, 1.0).detach()

    total = render_lum.sum() * 0.0
    used = 0
    for scale in scales:
        scale = int(scale)
        if scale <= 1:
            pred_s, gt_s, rel_s = render_lum, gt_lum, rel
        else:
            pred_s = F.avg_pool2d(render_lum, kernel_size=scale, stride=scale)
            gt_s = F.avg_pool2d(gt_lum, kernel_size=scale, stride=scale)
            rel_s = F.avg_pool2d(rel, kernel_size=scale, stride=scale)

        pred_gx, pred_gy = _sobel_xy(pred_s)
        gt_gx, gt_gy = _sobel_xy(gt_s)
        grad_diff = torch.sqrt((pred_gx - gt_gx).pow(2) + (pred_gy - gt_gy).pow(2) + eps * eps)

        gt_edge = torch.sqrt(gt_gx.detach().pow(2) + gt_gy.detach().pow(2) + eps * eps)
        edge_norm = gt_edge / gt_edge.mean().clamp_min(eps)
        edge_weight = 1.0 + max(float(edge_boost), 0.0) * edge_norm.clamp(0.0, 1.5)
        weight = rel_s.pow(max(float(reliability_power), 0.0)) * edge_weight
        min_reliability = min(max(float(min_reliability), 0.0), 1.0)
        if min_reliability > 0.0:
            weight = torch.where(rel_s >= min_reliability, weight, torch.zeros_like(weight))
        total = total + (grad_diff * weight).sum() / weight.sum().clamp_min(1e-6)
        used += 1
    return total / max(used, 1)


def reliability_weighted_photometric_loss(render_image: torch.Tensor,
                                          gt_image: torch.Tensor,
                                          reliability: torch.Tensor = None,
                                          reliability_power: float = 1.0,
                                          min_reliability: float = 0.0,
                                          edge_boost: float = 1.0,
                                          residual_stop: float = 0.35,
                                          residual_floor: float = 0.0,
                                          loss_type: str = "charbonnier",
                                          eps: float = 1e-3) -> torch.Tensor:
    """
    Charbonnier RGB loss concentrated on reliable vehicle-detail pixels.

    This complements the global L1/SSIM loss with a PSNR-oriented local signal.
    The detached residual gate prevents very large early errors from dominating,
    while GT edges raise the weight on contours and decals that matter for SSIM.
    """
    pred = _as_nchw(render_image).float()
    gt = _as_nchw(gt_image).float()
    if reliability is None:
        rel = torch.ones_like(pred[:, :1])
    else:
        rel = reliability.float()
        if rel.dim() == 2:
            rel = rel[None, None]
        elif rel.dim() == 3:
            rel = rel[None] if rel.shape[0] != 1 else rel[:, None]
        rel = rel.to(pred.device).clamp(0.0, 1.0).detach()

    gt_lum = _luminance(gt)
    edge_x, edge_y = _sobel_xy(gt_lum)
    edge = torch.sqrt(edge_x.detach().pow(2) + edge_y.detach().pow(2) + eps * eps)
    edge = edge / edge.mean().clamp_min(eps)
    edge_weight = 1.0 + max(float(edge_boost), 0.0) * edge.clamp(0.0, 1.5)

    residual = (pred.detach() - gt.detach()).abs().mean(dim=1, keepdim=True)
    residual_stop = max(float(residual_stop), eps)
    residual_floor = min(max(float(residual_floor), 0.0), 1.0)
    residual_gate = torch.exp(-residual / residual_stop).clamp(0.0, 1.0)
    if residual_floor > 0.0:
        residual_gate = residual_floor + (1.0 - residual_floor) * residual_gate
    weight = rel.pow(max(float(reliability_power), 0.0)) * edge_weight * residual_gate
    min_reliability = min(max(float(min_reliability), 0.0), 1.0)
    if min_reliability > 0.0:
        weight = torch.where(rel >= min_reliability, weight, torch.zeros_like(weight))

    residual_rgb = pred - gt
    loss_type = str(loss_type).lower()
    if loss_type in ("l2", "mse"):
        diff = residual_rgb.pow(2)
    elif loss_type == "l1":
        diff = residual_rgb.abs()
    else:
        diff = torch.sqrt(residual_rgb.pow(2) + eps * eps)
    return (diff * weight).sum() / (weight.sum().clamp_min(1e-6) * pred.shape[1])


def flat_region_mse_loss(render_image: torch.Tensor,
                         gt_image: torch.Tensor,
                         reliability: torch.Tensor = None,
                         edge_suppression: float = 1.5,
                         min_weight: float = 0.15,
                         residual_gain: float = 1.0,
                         residual_kernel: int = 7,
                         reliability_power: float = 0.0,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    PSNR-oriented MSE that avoids over-penalizing high-frequency structure.

    Global MSE improves pixel accuracy but can smooth contours and decals. This
    variant raises the MSE signal on low-edge regions while keeping enough floor
    weight elsewhere to avoid ignoring the vehicle body entirely.
    """
    pred = _as_nchw(render_image).float()
    gt = _as_nchw(gt_image).float()

    gt_lum = _luminance(gt)
    edge_x, edge_y = _sobel_xy(gt_lum)
    edge = torch.sqrt(edge_x.detach().pow(2) + edge_y.detach().pow(2) + eps)
    edge = edge / edge.mean().clamp_min(eps)
    flat_weight = torch.exp(-max(float(edge_suppression), 0.0) * edge.clamp(0.0, 3.0))
    min_weight = min(max(float(min_weight), 0.0), 1.0)
    if min_weight > 0.0:
        flat_weight = min_weight + (1.0 - min_weight) * flat_weight

    residual_gain = max(float(residual_gain), 0.0)
    if residual_gain > 0.0:
        residual = (pred.detach() - gt.detach()).abs().mean(dim=1, keepdim=True)
        residual = local_mean(residual[0, 0], max(int(residual_kernel), 1))[None, None]
        residual = residual / residual.mean().clamp_min(eps)
        flat_weight = flat_weight * (1.0 + residual_gain * residual.clamp(0.0, 2.0))

    if reliability is not None and reliability_power > 0.0:
        rel = reliability.float()
        if rel.dim() == 2:
            rel = rel[None, None]
        elif rel.dim() == 3:
            rel = rel[None] if rel.shape[0] != 1 else rel[:, None]
        rel = rel.to(pred.device).clamp(0.0, 1.0).detach()
        flat_weight = flat_weight * rel.pow(max(float(reliability_power), 0.0))

    diff = (pred - gt).pow(2)
    return (diff * flat_weight).sum() / (flat_weight.sum().clamp_min(eps) * pred.shape[1])


def reliability_weighted_ssim_loss(render_image: torch.Tensor,
                                   gt_image: torch.Tensor,
                                   reliability: torch.Tensor = None,
                                   window_size: int = 11,
                                   reliability_power: float = 1.0,
                                   min_reliability: float = 0.0,
                                   edge_boost: float = 0.5,
                                   eps: float = 1e-6) -> torch.Tensor:
    """
    Local SSIM loss concentrated on reliable vehicle-detail pixels.

    The global DSSIM term optimizes the whole image uniformly. This variant uses
    the same SSIM statistics, but weights each local window by reliability and GT
    edge strength so structural improvements on the car are not diluted by easy
    background pixels.
    """
    pred = _as_nchw(render_image).float()
    gt = _as_nchw(gt_image).float()
    channel = pred.shape[1]
    window_size = max(int(window_size), 3)
    if window_size % 2 == 0:
        window_size += 1

    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    gaussian = torch.exp(-(coords * coords) / (2.0 * 1.5 * 1.5))
    gaussian = gaussian / gaussian.sum().clamp_min(eps)
    window = (gaussian[:, None] @ gaussian[None, :]).view(1, 1, window_size, window_size)
    window = window.expand(channel, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_pred = F.conv2d(pred, window, padding=pad, groups=channel)
    mu_gt = F.conv2d(gt, window, padding=pad, groups=channel)
    mu_pred_sq = mu_pred.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_pred_gt = mu_pred * mu_gt
    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, window, padding=pad, groups=channel) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, window, padding=pad, groups=channel) - mu_pred_gt

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu_pred_gt + c1) * (2.0 * sigma_pred_gt + c2)) / (
        (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2) + eps
    )
    ssim_map = ssim_map.mean(dim=1, keepdim=True).clamp(-1.0, 1.0)

    if reliability is None:
        rel = torch.ones_like(ssim_map)
    else:
        rel = reliability.float()
        if rel.dim() == 2:
            rel = rel[None, None]
        elif rel.dim() == 3:
            rel = rel[None] if rel.shape[0] != 1 else rel[:, None]
        rel = rel.to(pred.device).clamp(0.0, 1.0).detach()

    gt_lum = _luminance(gt)
    edge_x, edge_y = _sobel_xy(gt_lum)
    edge = torch.sqrt(edge_x.detach().pow(2) + edge_y.detach().pow(2) + eps)
    edge = edge / edge.mean().clamp_min(eps)
    edge_weight = 1.0 + max(float(edge_boost), 0.0) * edge.clamp(0.0, 1.5)

    weight = rel.pow(max(float(reliability_power), 0.0)) * edge_weight
    min_reliability = min(max(float(min_reliability), 0.0), 1.0)
    if min_reliability > 0.0:
        weight = torch.where(rel >= min_reliability, weight, torch.zeros_like(weight))
    return ((1.0 - ssim_map) * weight).sum() / weight.sum().clamp_min(eps)


def reliability_weighted_contrast_loss(render_image: torch.Tensor,
                                       gt_image: torch.Tensor,
                                       reliability: torch.Tensor = None,
                                       window_size: int = 9,
                                       scales=(1, 2, 4),
                                       reliability_power: float = 1.0,
                                       min_reliability: float = 0.0,
                                       edge_boost: float = 0.5,
                                       mean_weight: float = 0.25,
                                       eps: float = 1e-4) -> torch.Tensor:
    """
    Reliable local contrast consistency for SSIM/LPIPS-sensitive details.

    SSIM rewards local luminance and contrast agreement. The normal local SSIM
    loss can become flat once structures are roughly aligned, so this term adds a
    direct signal on local mean and standard deviation in reliable edge/detail
    regions.
    """
    pred_lum = _luminance(render_image)
    gt_lum = _luminance(gt_image)
    if reliability is None:
        rel = torch.ones_like(pred_lum[:, :1])
    else:
        rel = reliability.float()
        if rel.dim() == 2:
            rel = rel[None, None]
        elif rel.dim() == 3:
            rel = rel[None] if rel.shape[0] != 1 else rel[:, None]
        rel = rel.to(pred_lum.device).clamp(0.0, 1.0).detach()

    window_size = max(int(window_size), 3)
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    mean_weight = max(float(mean_weight), 0.0)
    min_reliability = min(max(float(min_reliability), 0.0), 1.0)

    total = pred_lum.sum() * 0.0
    used = 0
    for scale in scales:
        scale = int(scale)
        if scale <= 1:
            pred_s, gt_s, rel_s = pred_lum, gt_lum, rel
        else:
            pred_s = F.avg_pool2d(pred_lum, kernel_size=scale, stride=scale)
            gt_s = F.avg_pool2d(gt_lum, kernel_size=scale, stride=scale)
            rel_s = F.avg_pool2d(rel, kernel_size=scale, stride=scale)

        pred_mean = F.avg_pool2d(F.pad(pred_s, (pad, pad, pad, pad), mode="reflect"), window_size, stride=1)
        gt_mean = F.avg_pool2d(F.pad(gt_s, (pad, pad, pad, pad), mode="reflect"), window_size, stride=1)
        pred_sq_mean = F.avg_pool2d(F.pad(pred_s * pred_s, (pad, pad, pad, pad), mode="reflect"), window_size, stride=1)
        gt_sq_mean = F.avg_pool2d(F.pad(gt_s * gt_s, (pad, pad, pad, pad), mode="reflect"), window_size, stride=1)
        pred_std = (pred_sq_mean - pred_mean.pow(2)).clamp_min(0.0).sqrt()
        gt_std = (gt_sq_mean - gt_mean.pow(2)).clamp_min(0.0).sqrt()

        gt_gx, gt_gy = _sobel_xy(gt_s)
        edge = torch.sqrt(gt_gx.detach().pow(2) + gt_gy.detach().pow(2) + eps)
        edge = edge / edge.mean().clamp_min(eps)
        edge_weight = 1.0 + max(float(edge_boost), 0.0) * edge.clamp(0.0, 1.5)
        weight = rel_s.pow(max(float(reliability_power), 0.0)) * edge_weight
        if min_reliability > 0.0:
            weight = torch.where(rel_s >= min_reliability, weight, torch.zeros_like(weight))

        contrast_loss = torch.abs(pred_std - gt_std)
        luminance_loss = torch.abs(pred_mean - gt_mean)
        total = total + ((contrast_loss + mean_weight * luminance_loss) * weight).sum() / weight.sum().clamp_min(eps)
        used += 1
    return total / max(used, 1)


class SurfacePriorMemory:
    """
    Lightweight self-constrained depth prior.

    This is a training-time analogue of GSPrior's self-updating surface prior.
    Instead of building a full TSDF volume in the first version, it stores a
    reliable low-resolution rendered depth memory for each training view.
    """

    def __init__(self, downsample: int = 4, momentum: float = 0.95,
                 min_reliability: float = 0.10, warmup: int = 1500,
                 start_iter: int = 2000):
        self.downsample = max(1, int(downsample))
        self.momentum = float(momentum)
        self.min_reliability = float(min_reliability)
        self.warmup = int(warmup)
        self.start_iter = int(start_iter)
        self.depth = {}
        self.weight = {}

    def _key(self, camera) -> str:
        return getattr(camera, "image_name", str(getattr(camera, "uid", id(camera))))

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x[None, None]
        elif x.dim() == 3:
            x = x[None]
        if self.downsample > 1:
            x = F.avg_pool2d(x.float(), kernel_size=self.downsample, stride=self.downsample)
        return x[0]

    @torch.no_grad()
    def update(self, camera, surf_depth: torch.Tensor, reliability: torch.Tensor, iteration: int):
        key = self._key(camera)
        depth_ds = self._resize(surf_depth.detach())
        weight_ds = self._resize(reliability.detach()).clamp(0.0, 1.0)
        valid = (depth_ds > 0).float() * (weight_ds >= self.min_reliability).float()
        weight_ds = weight_ds * valid
        if weight_ds.sum() < 1:
            return
        if key not in self.depth or iteration < self.warmup:
            self.depth[key] = depth_ds.clone()
            self.weight[key] = weight_ds.clone()
            return
        old_depth = self.depth[key]
        old_weight = self.weight[key]
        update_weight = weight_ds
        merged_weight = torch.maximum(old_weight * self.momentum, update_weight)
        self.depth[key] = self.momentum * old_depth + (1.0 - self.momentum) * depth_ds
        self.weight[key] = merged_weight.clamp(0.0, 1.0)

    def loss(self, camera, surf_depth: torch.Tensor, reliability: torch.Tensor, iteration: int,
             eps: float = 1e-3) -> torch.Tensor:
        key = self._key(camera)
        if iteration < self.start_iter or key not in self.depth:
            return surf_depth.sum() * 0.0
        depth_ds = self._resize(surf_depth)
        rel_ds = self._resize(reliability).clamp(0.0, 1.0)
        prior_depth = self.depth[key].to(depth_ds.device)
        prior_weight = self.weight[key].to(depth_ds.device)
        weight = (prior_weight * rel_ds).detach()
        valid = (depth_ds > 0).float() * (prior_depth > 0).float()
        weight = weight * valid
        denom = weight.sum().clamp_min(1e-6)
        diff = torch.sqrt((depth_ds - prior_depth.detach()).pow(2) + eps * eps)
        return (diff * weight).sum() / denom
