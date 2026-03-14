import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import _list_image_files

try:
    import timm
except Exception as exc:
    raise RuntimeError("timm is required for perceptual evaluation.") from exc


def _pil_to_tensor(img):
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _center_crop(img, size):
    w, h = img.size
    if w < size or h < size:
        scale = max(size / max(1, w), size / max(1, h))
        nw = max(size, int(round(w * scale)))
        nh = max(size, int(round(h * scale)))
        img = img.resize((nw, nh), resample=Image.Resampling.BICUBIC)
        w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _random_resized_crop(img, size, scale_min=0.7, scale_max=1.0):
    w, h = img.size
    area = w * h
    for _ in range(10):
        target = random.uniform(scale_min, scale_max) * area
        ratio = random.uniform(0.85, 1.2)
        nw = int(round((target * ratio) ** 0.5))
        nh = int(round((target / ratio) ** 0.5))
        if 0 < nw <= w and 0 < nh <= h:
            x = random.randint(0, w - nw)
            y = random.randint(0, h - nh)
            crop = img.crop((x, y, x + nw, y + nh))
            return crop.resize((size, size), resample=Image.Resampling.BICUBIC)
    return _center_crop(img.resize((size, size), resample=Image.Resampling.BICUBIC), size)


def _normalize(x, mean, std):
    mean_t = torch.tensor(mean, dtype=x.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(3, 1, 1)
    return (x - mean_t) / std_t


def _weak_aug(img, size):
    out = _random_resized_crop(img, size=size)
    if random.random() < 0.5:
        out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.6:
        b = ImageEnhance.Brightness(out)
        out = b.enhance(random.uniform(0.85, 1.15))
        c = ImageEnhance.Contrast(out)
        out = c.enhance(random.uniform(0.85, 1.15))
        s = ImageEnhance.Color(out)
        out = s.enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.4:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.8)))
    return out


class PerceptualEvalDataset(Dataset):
    def __init__(self, root, exts, image_size, mean, std, max_images=0):
        paths = _list_image_files(root, exts)
        if not paths:
            raise ValueError(f"No images found in: {root}")
        if max_images and max_images > 0:
            paths = paths[:max_images]
        self.paths = paths
        self.image_size = int(image_size)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        clean_img = _center_crop(img.resize((self.image_size, self.image_size), resample=Image.Resampling.BICUBIC), self.image_size)
        view1_img = _weak_aug(img, self.image_size)
        view2_img = _weak_aug(img, self.image_size)

        raw = _pil_to_tensor(clean_img)
        clean = _normalize(raw.clone(), self.mean, self.std)
        view1 = _normalize(_pil_to_tensor(view1_img), self.mean, self.std)
        view2 = _normalize(_pil_to_tensor(view2_img), self.mean, self.std)
        return clean, view1, view2, raw


def build_backbone(model_name, checkpoint_path, device):
    model = timm.create_model(model_name, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "backbone" in state:
        state = state["backbone"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    return model


def extract_features(backbone, x):
    feat = backbone.forward_features(x)
    if feat.ndim == 4:
        feat = feat.mean(dim=(-2, -1))
    return F.normalize(feat.float(), dim=1)


def distort_tensor_batch(x, severity):
    # x is [B,3,H,W] in [0,1]
    blur_sigmas = {1: 0.5, 2: 1.2, 3: 2.0}
    noise_sigmas = {1: 0.01, 2: 0.03, 3: 0.06}
    k_sizes = {1: 3, 2: 5, 3: 7}

    sigma = blur_sigmas[severity]
    noise = noise_sigmas[severity]
    k = k_sizes[severity]

    # Depthwise Gaussian blur with a separable approximation.
    radius = k // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)

    out = F.conv2d(x, kernel_2d, padding=radius, groups=x.shape[1])
    out = torch.clamp(out + torch.randn_like(out) * noise, 0.0, 1.0)
    return out


def normalize_batch(x, mean, std):
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def compute_uniformity(z, pairs=20000, seed=42):
    n = z.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    zi = z[i]
    zj = z[j]
    dist2 = ((zi - zj) ** 2).sum(dim=1)
    uniformity = torch.log(torch.exp(-2.0 * dist2).mean() + 1e-12)
    return float(uniformity.cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="convnextv2_base.fcmae_ft_in22k_in1k")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--extensions", nargs="+", default=["png", "jpg", "jpeg", "webp"])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dataset = PerceptualEvalDataset(
        root=args.val_dir,
        exts=args.extensions,
        image_size=args.image_size,
        mean=mean,
        std=std,
        max_images=args.max_images,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    backbone = build_backbone(args.model_name, args.checkpoint, device)

    feats_clean = []
    feats_v1 = []
    feats_v2 = []
    d1_all = []
    d2_all = []
    d3_all = []

    with torch.inference_mode():
        for clean, view1, view2, raw in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            clean = clean.to(device, non_blocking=True)
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            raw = raw.to(device, non_blocking=True)

            z_clean = extract_features(backbone, clean)
            z1 = extract_features(backbone, view1)
            z2 = extract_features(backbone, view2)

            feats_clean.append(z_clean.cpu())
            feats_v1.append(z1.cpu())
            feats_v2.append(z2.cpu())

            raw_d1 = distort_tensor_batch(raw, severity=1)
            raw_d2 = distort_tensor_batch(raw, severity=2)
            raw_d3 = distort_tensor_batch(raw, severity=3)

            d1 = extract_features(backbone, normalize_batch(raw_d1, mean, std))
            d2 = extract_features(backbone, normalize_batch(raw_d2, mean, std))
            d3 = extract_features(backbone, normalize_batch(raw_d3, mean, std))

            dist1 = (1.0 - (z_clean * d1).sum(dim=1)).cpu()
            dist2 = (1.0 - (z_clean * d2).sum(dim=1)).cpu()
            dist3 = (1.0 - (z_clean * d3).sum(dim=1)).cpu()

            d1_all.append(dist1)
            d2_all.append(dist2)
            d3_all.append(dist3)

    feats_clean = torch.cat(feats_clean, dim=0)
    feats_v1 = torch.cat(feats_v1, dim=0)
    feats_v2 = torch.cat(feats_v2, dim=0)

    sim = feats_v1 @ feats_v2.T
    n = sim.shape[0]
    diag = sim.diag()
    top1 = sim.argmax(dim=1)
    retrieval_at1 = (top1 == torch.arange(n)).float().mean().item()

    if n > 1:
        neg_sum = sim.sum() - diag.sum()
        neg_count = n * n - n
        neg_mean = (neg_sum / neg_count).item()
    else:
        neg_mean = float("nan")

    pos_mean = diag.mean().item()
    gap = pos_mean - neg_mean if n > 1 else float("nan")

    alignment_l2 = ((feats_v1 - feats_v2) ** 2).sum(dim=1).mean().item()
    uniformity = compute_uniformity(feats_clean, pairs=20000, seed=args.seed)

    d1 = torch.cat(d1_all, dim=0)
    d2 = torch.cat(d2_all, dim=0)
    d3 = torch.cat(d3_all, dim=0)
    monotonic = ((d1 < d2) & (d2 < d3)).float().mean().item()

    result = {
        "num_images": int(n),
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "image_size": int(args.image_size),
        "retrieval_at1": float(retrieval_at1),
        "pos_cosine_mean": float(pos_mean),
        "neg_cosine_mean": float(neg_mean),
        "pos_neg_gap": float(gap),
        "alignment_l2": float(alignment_l2),
        "uniformity": float(uniformity),
        "distortion_dist_mean_s1": float(d1.mean().item()),
        "distortion_dist_mean_s2": float(d2.mean().item()),
        "distortion_dist_mean_s3": float(d3.mean().item()),
        "distortion_monotonic_ratio": float(monotonic),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
