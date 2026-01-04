import torch
import torch.nn.functional as F

try:
    from dists_pytorch import DISTS
except Exception:
    try:
        from DISTS_pytorch import DISTS
    except Exception:
        DISTS = None


def build_dists(device):
    if DISTS is None:
        return None
    model = DISTS().to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    window = window.view(1, 1, window_size, window_size)
    return window.repeat(channels, 1, 1, 1)


@torch.no_grad()
def calc_psnr(pred, target, eps=1e-10):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr


@torch.no_grad()
def calc_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    if pred.ndim != 4:
        raise ValueError("Expected BCHW input for SSIM.")
    _, channels, _, _ = pred.shape
    dtype = pred.dtype
    device = pred.device
    window = _gaussian_window(window_size, sigma, channels, device, dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean(dim=(1, 2, 3))


@torch.no_grad()
def calc_dists(dists_model, pred, target):
    if dists_model is None:
        return None
    pred = pred.float()
    target = target.float()
    score = dists_model(pred, target)
    if score.ndim == 0:
        score = score.view(1)
    return score
