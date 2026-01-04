import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import VGG19_Weights, vgg19
except Exception:
    VGG19_Weights = None
    vgg19 = None


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        pred_dx = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred_gray, self.sobel_y, padding=1)
        tgt_dx = F.conv2d(target_gray, self.sobel_x, padding=1)
        tgt_dy = F.conv2d(target_gray, self.sobel_y, padding=1)
        pred_mag = torch.sqrt(pred_dx * pred_dx + pred_dy * pred_dy + 1e-6)
        tgt_mag = torch.sqrt(tgt_dx * tgt_dx + tgt_dy * tgt_dy + 1e-6)
        return F.l1_loss(pred_mag, tgt_mag)


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        if pred.dtype not in (torch.float32, torch.float64):
            pred = pred.float()
        if target.dtype not in (torch.float32, torch.float64):
            target = target.float()
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_ids, use_pretrained=True):
        super().__init__()
        if vgg19 is None:
            raise RuntimeError("torchvision is required for perceptual loss.")
        weights = VGG19_Weights.DEFAULT if use_pretrained else None
        vgg = vgg19(weights=weights).features
        max_id = max(layer_ids)
        self.features = vgg[: max_id + 1].eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.layer_ids = set(layer_ids)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_ids:
                feats.append(x)
        return feats


class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids, layer_weights=None, use_pretrained=True):
        super().__init__()
        self.extractor = VGGFeatureExtractor(layer_ids, use_pretrained=use_pretrained)
        if layer_weights is None:
            self.layer_weights = [1.0] * len(layer_ids)
        else:
            self.layer_weights = layer_weights
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        pred_feats = self.extractor(pred)
        tgt_feats = self.extractor(target)
        loss = 0.0
        for w, pf, tf in zip(self.layer_weights, pred_feats, tgt_feats):
            loss = loss + w * F.l1_loss(pf, tf)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pixel_type = cfg.get("pixel_type", "charbonnier")
        if pixel_type == "l1":
            self.pixel_loss = nn.L1Loss()
        else:
            self.pixel_loss = CharbonnierLoss(eps=float(cfg.get("charbonnier_eps", 1e-3)))

        self.pixel_weight = float(cfg.get("pixel_weight", 1.0))
        self.edge_weight = float(cfg.get("edge_weight", 0.0))
        self.fft_weight = float(cfg.get("fft_weight", 0.0))
        self.perceptual_weight = float(cfg.get("perceptual_weight", 0.0))

        self.edge_loss = EdgeLoss() if self.edge_weight > 0 else None
        self.fft_loss = FFTLoss() if self.fft_weight > 0 else None

        self.perceptual_loss = None
        if self.perceptual_weight > 0:
            layer_ids = cfg.get("perceptual_layers", [8, 17, 26])
            layer_weights = cfg.get("perceptual_layer_weights", [1.0, 1.0, 1.0])
            use_pretrained = bool(cfg.get("perceptual_pretrained", True))
            self.perceptual_loss = PerceptualLoss(layer_ids, layer_weights, use_pretrained=use_pretrained)

    def forward(self, pred, target):
        total = 0.0
        details = {}
        if self.pixel_weight > 0:
            pixel = self.pixel_loss(pred, target) * self.pixel_weight
            total = total + pixel
            details["pixel"] = float(pixel.detach().cpu())
        if self.edge_loss is not None:
            edge = self.edge_loss(pred, target) * self.edge_weight
            total = total + edge
            details["edge"] = float(edge.detach().cpu())
        if self.fft_loss is not None:
            freq = self.fft_loss(pred, target) * self.fft_weight
            total = total + freq
            details["fft"] = float(freq.detach().cpu())
        if self.perceptual_loss is not None:
            perc = self.perceptual_loss(pred, target) * self.perceptual_weight
            total = total + perc
            details["perceptual"] = float(perc.detach().cpu())
        details["total"] = float(total.detach().cpu())
        return total, details
