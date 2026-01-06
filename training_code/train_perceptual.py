import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from data import _list_image_files
from utils import append_jsonl, ensure_dir, load_config, set_seed

try:
    import timm
except Exception as exc:
    raise RuntimeError("timm is required for ConvNeXtV2 perceptual pretraining.") from exc


class AnimeImageDataset(Dataset):
    def __init__(self, root, exts, transform):
        self.paths = _list_image_files(root, exts)
        if not self.paths:
            raise ValueError(f"No images found in: {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        last_err = None
        for _ in range(5):
            path = self.paths[idx]
            try:
                img = Image.open(path).convert("RGB")
                view1 = self.transform(img)
                view2 = self.transform(img)
                return view1, view2
            except Exception as err:
                last_err = err
                idx = random.randint(0, len(self.paths) - 1)
        raise RuntimeError(f"Failed to load image after retries: {self.paths[idx]}") from last_err


class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name, use_pretrained):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=use_pretrained)

    def forward(self, x):
        feat = self.model.forward_features(x)
        if feat.ndim == 4:
            feat = feat.mean(dim=(-2, -1))
        return feat


def _mlp(in_dim, hidden_dim, out_dim, num_layers):
    if num_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SimSiam(nn.Module):
    def __init__(self, backbone, feature_dim, proj_dim, pred_dim):
        super().__init__()
        self.backbone = backbone
        self.projector = _mlp(feature_dim, proj_dim, proj_dim, num_layers=3)
        self.predictor = _mlp(proj_dim, pred_dim, proj_dim, num_layers=2)

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def build_scheduler(optimizer, cfg, total_steps):
    name = str(cfg.get("name", "cosine")).lower()
    warmup_steps = int(cfg.get("warmup_steps", 0))
    min_lr = float(cfg.get("min_lr", 0.0))
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if name == "constant":
            return 1.0
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_transforms(image_size, mean, std):
    blur_kernel = int(max(3, (image_size // 20) | 1))
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _resolve_compile_backend(compile_cfg):
    backend = str(compile_cfg.get("backend", "inductor")).lower()
    fallback = str(compile_cfg.get("fallback_backend", "aot_eager")).strip().lower()
    if backend != "inductor":
        return backend
    try:
        from triton.compiler import compiler as triton_compiler

        if not hasattr(triton_compiler, "triton_key"):
            raise ImportError("triton_key is missing")
    except Exception as exc:
        if fallback:
            print(f"Inductor backend unavailable ({exc}); using {fallback}.")
            return fallback
        print(f"Inductor backend unavailable ({exc}); skipping torch.compile.")
        return None
    return backend


def maybe_compile(model, compile_cfg, device):
    if not isinstance(compile_cfg, dict) or not bool(compile_cfg.get("enabled", False)):
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile is not available; skipping.")
        return model
    if device.type != "cuda":
        print("torch.compile is enabled but CUDA is not available; skipping.")
        return model
    backend = _resolve_compile_backend(compile_cfg)
    if backend is None:
        return model
    mode = str(compile_cfg.get("mode", "default"))
    fullgraph = bool(compile_cfg.get("fullgraph", False))
    dynamic = bool(compile_cfg.get("dynamic", False))
    suppress_errors = bool(compile_cfg.get("suppress_errors", False))
    if suppress_errors:
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except Exception as exc:
            print(f"Warning: failed to set torch._dynamo.config.suppress_errors: {exc}")
    try:
        model = torch.compile(
            model,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
    except Exception as exc:
        fallback = str(compile_cfg.get("fallback_backend", "")).strip().lower()
        if fallback and fallback != backend:
            print(f"torch.compile failed with backend={backend}: {exc}. Trying {fallback}.")
            try:
                model = torch.compile(
                    model,
                    backend=fallback,
                    mode=mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
                print(
                    "torch.compile enabled "
                    f"(backend={fallback}, mode={mode}, fullgraph={fullgraph}, dynamic={dynamic})."
                )
                return model
            except Exception as fallback_exc:
                print(f"torch.compile failed: {fallback_exc}. Falling back to eager.")
                return model
        print(f"torch.compile failed: {exc}. Falling back to eager.")
        return model
    print(
        "torch.compile enabled "
        f"(backend={backend}, mode={mode}, fullgraph={fullgraph}, dynamic={dynamic})."
    )
    return model


def _resolve_model_name(cfg):
    model_name = cfg.get("model_name")
    if model_name:
        return model_name

    variant = str(cfg.get("variant", "tiny")).lower()
    if variant == "base":
        return "convnextv2_base.fcmae_ft_in22k_in1k"
    if variant == "tiny":
        return "convnextv2_tiny.fcmae_ft_in22k_in1k"
    raise ValueError(f"Unsupported ConvNeXtV2 variant: {variant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("perceptual_train", {})
    set_seed(train_cfg.get("seed", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg.get("data", {})
    data_dir = train_cfg.get("data_dir") or data_cfg.get("train_dir")
    if not data_dir:
        raise ValueError("perceptual_train.data_dir or data.train_dir is required")

    image_size = int(train_cfg.get("image_size", 224))
    mean = train_cfg.get("mean", [0.485, 0.456, 0.406])
    std = train_cfg.get("std", [0.229, 0.224, 0.225])
    transform = build_transforms(image_size, mean, std)

    dataset = AnimeImageDataset(data_dir, data_cfg.get("extensions", ["png", "jpg", "jpeg", "webp"]), transform)
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 6))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    use_pretrained = bool(train_cfg.get("pretrained", False))
    model_name = _resolve_model_name(train_cfg)
    backbone = ConvNeXtBackbone(model_name=model_name, use_pretrained=use_pretrained)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        feat_dim = backbone(dummy).shape[1]

    proj_dim = int(train_cfg.get("proj_dim", 2048))
    pred_dim = int(train_cfg.get("pred_dim", 512))
    model = SimSiam(backbone, feature_dim=feat_dim, proj_dim=proj_dim, pred_dim=pred_dim).to(device)
    base_model = model

    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    epochs = int(train_cfg.get("epochs", 100))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every_epochs", 1))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))

    total_steps = max(1, len(loader) * epochs)
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}), total_steps=total_steps)

    amp_mode = str(train_cfg.get("amp", "bf16")).lower()
    use_amp = amp_mode in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(enabled=(amp_mode == "fp16" and device.type == "cuda"))

    output_dir = Path(train_cfg.get("output_dir", "runs/perceptual_convnextv2"))
    ensure_dir(output_dir)
    log_path = output_dir / "train_log.jsonl"

    start_epoch = 0
    global_step = 0
    resume_path = train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        base_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {resume_path} at epoch {start_epoch + 1}")

    model = maybe_compile(base_model, train_cfg.get("compile", {}), device)

    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for view1, view2 in progress:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            if use_amp:
                ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                ctx = nullcontext()

            with ctx:
                p1, p2, z1, z2 = model(view1, view2)
                loss = 0.5 * (simsiam_loss(p1, z2) + simsiam_loss(p2, z1))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            if global_step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                append_jsonl(log_path, {"epoch": epoch + 1, "step": global_step, "lr": lr_now, "loss": float(loss)})
            progress.set_postfix(loss=float(loss))

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            if scaler.is_enabled():
                ckpt["scaler"] = scaler.state_dict()

            ckpt_path = output_dir / f"ckpt_epoch_{epoch + 1:04d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, output_dir / "ckpt_last.pt")

            backbone_state = base_model.backbone.model.state_dict()
            torch.save({"backbone": backbone_state}, output_dir / f"convnextv2_backbone_{epoch + 1:04d}.pt")
            torch.save({"backbone": backbone_state}, output_dir / "convnextv2_backbone_last.pt")


if __name__ == "__main__":
    main()
