import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, c)


def _window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(n, n, -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, h, w):
        b, l, c = x.shape
        if l != h * w:
            raise ValueError("Input feature has wrong size")
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._build_attn_mask(hp, wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = _window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_h > 0 or pad_w > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _build_attn_mask(self, h, w, device):
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = _window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class ResidualSwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio,
        qkv_bias,
        drop,
        attn_drop,
        drop_path,
    ):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, h, w):
        residual = x
        for blk in self.blocks:
            x = blk(x, h, w)
        b, l, c = x.shape
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, l, c)
        return x + residual


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if scale in (2, 3):
            m.append(nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1))
            m.append(nn.PixelShuffle(scale))
        elif scale == 4:
            for _ in range(2):
                m.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*m)


class AnimeSwinIR(nn.Module):
    def __init__(
        self,
        scale=2,
        in_channels=3,
        embed_dim=96,
        depths=(6, 6, 6, 6),
        num_heads=(6, 6, 6, 6),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.scale = scale
        self.window_size = window_size
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_idx = 0
        for i in range(len(depths)):
            layer = ResidualSwinBlock(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[dpr_idx : dpr_idx + depths[i]],
            )
            dpr_idx += depths[i]
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = Upsample(scale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_first(x)
        res = x
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, h, w)
        x = self.norm(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv_after_body(x) + res
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)
        return x


def build_model(cfg, scale):
    return AnimeSwinIR(
        scale=scale,
        in_channels=int(cfg.get("in_channels", 3)),
        embed_dim=int(cfg.get("embed_dim", 96)),
        depths=tuple(cfg.get("depths", (6, 6, 6, 6))),
        num_heads=tuple(cfg.get("num_heads", (6, 6, 6, 6))),
        window_size=int(cfg.get("window_size", 8)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        qkv_bias=bool(cfg.get("qkv_bias", True)),
        drop_rate=float(cfg.get("drop_rate", 0.0)),
        attn_drop_rate=float(cfg.get("attn_drop_rate", 0.0)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.1)),
    )
