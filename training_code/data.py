import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset


_RESAMPLE_MAP = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}


def _list_image_files(root, exts):
    root = Path(root)
    paths = []
    for ext in exts:
        paths.extend(root.rglob(f"*.{ext}"))
    return sorted(paths)


def _pil_to_tensor(img):
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class DegradationPipeline:
    def __init__(self, scale, cfg):
        self.scale = scale
        self.blur_prob = float(cfg.get("blur_prob", 0.2))
        self.blur_radius = float(cfg.get("blur_radius", 1.2))
        self.noise_prob = float(cfg.get("noise_prob", 0.2))
        self.noise_sigma = float(cfg.get("noise_sigma", 2.0))
        self.jpeg_prob = float(cfg.get("jpeg_prob", 0.2))
        self.jpeg_quality_min = int(cfg.get("jpeg_quality_min", 70))
        self.jpeg_quality_max = int(cfg.get("jpeg_quality_max", 95))
        self.resample_choices = cfg.get("resample_methods", ["bicubic"])

    def _random_resample(self):
        key = random.choice(self.resample_choices)
        return _RESAMPLE_MAP.get(key, Image.Resampling.BICUBIC)

    def _apply_noise(self, img):
        if self.noise_prob <= 0 or random.random() >= self.noise_prob:
            return img
        arr = np.array(img, dtype=np.float32)
        sigma = random.uniform(0.1, self.noise_sigma)
        noise = np.random.normal(0.0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def _apply_jpeg(self, img):
        if self.jpeg_prob <= 0 or random.random() >= self.jpeg_prob:
            return img
        quality = random.randint(self.jpeg_quality_min, self.jpeg_quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, subsampling=0, optimize=True)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __call__(self, hr_img):
        if self.blur_prob > 0 and random.random() < self.blur_prob:
            radius = random.uniform(0.1, self.blur_radius)
            hr_img = hr_img.filter(ImageFilter.GaussianBlur(radius=radius))

        lr_w = max(1, hr_img.width // self.scale)
        lr_h = max(1, hr_img.height // self.scale)
        hr_img = hr_img.resize((lr_w, lr_h), resample=self._random_resample())
        hr_img = self._apply_noise(hr_img)
        hr_img = self._apply_jpeg(hr_img)
        return hr_img


class AnimeSRDataset(Dataset):
    def __init__(self, root, scale, hr_patch_size, exts, augment, degrade_cfg, training=True, crop_to_scale=False):
        self.paths = _list_image_files(root, exts)
        if not self.paths:
            raise ValueError(f"No images found in: {root}")
        self.scale = int(scale)
        self.hr_patch_size = int(hr_patch_size)
        self.augment = augment
        self.training = training
        self.crop_to_scale = bool(crop_to_scale)
        self.degrade = DegradationPipeline(scale, degrade_cfg or {})

    def _random_crop(self, img):
        if self.hr_patch_size <= 0:
            return img
        patch = (self.hr_patch_size // self.scale) * self.scale
        patch = max(patch, self.scale)
        if img.width < patch or img.height < patch:
            img = img.resize((patch, patch), resample=Image.Resampling.BICUBIC)
            return img
        x = random.randint(0, img.width - patch)
        y = random.randint(0, img.height - patch)
        return img.crop((x, y, x + patch, y + patch))

    def _augment(self, img):
        if not self.augment:
            return img
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.ROTATE_90)
        return img

    def _center_crop_divisible(self, img):
        if self.scale <= 1:
            return img
        w, h = img.width, img.height
        new_w = (w // self.scale) * self.scale
        new_h = (h // self.scale) * self.scale
        if new_w == 0 or new_h == 0:
            return img
        if new_w == w and new_h == h:
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        last_err = None
        for _ in range(5):
            path = self.paths[idx]
            try:
                img = Image.open(path)
                if img.mode == "P":
                    img = img.convert("RGBA")
                img = img.convert("RGB")
                if not self.training and self.crop_to_scale:
                    img = self._center_crop_divisible(img)
                if self.training:
                    img = self._random_crop(img)
                    img = self._augment(img)
                lr = self.degrade(img)
                hr_tensor = _pil_to_tensor(img)
                lr_tensor = _pil_to_tensor(lr)
                return lr_tensor, hr_tensor
            except Exception as err:
                last_err = err
                idx = random.randint(0, len(self.paths) - 1)
        raise RuntimeError(f"Failed to load image after retries: {self.paths[idx]}") from last_err
