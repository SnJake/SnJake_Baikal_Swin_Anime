import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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
        relative_position_bias = relative_position_bias.view(n, n, -1).permute(2, 0, 1).to(dtype=attn.dtype)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n)
            attn = attn + mask.to(dtype=attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        if attn.dtype != v.dtype:
            attn = attn.to(dtype=v.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionV2(nn.Module):
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

        self.logit_scale = nn.Parameter(torch.log(torch.ones((num_heads, 1, 1)) * 10.0))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords_table = relative_coords.float()
        if window_size > 1:
            relative_coords_table[:, :, 0] /= (window_size - 1)
            relative_coords_table[:, :, 1] /= (window_size - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0
        ) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale

        relative_position_bias = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        relative_position_bias = relative_position_bias.to(dtype=attn.dtype)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n)
            attn = attn + mask.to(dtype=attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = F.softmax(attn, dim=-1)
        if attn.dtype != v.dtype:
            attn = attn.to(dtype=v.dtype)
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


class SwinTransformerBlockV2(nn.Module):
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
        self.attn = WindowAttentionV2(
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


class ResidualSwinBlockV2(nn.Module):
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
                SwinTransformerBlockV2(
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


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, h, w):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, h, w)


class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm="ortho"):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        input_dtype = x.dtype
        if input_dtype not in (torch.float32, torch.float64):
            x = x.float()
        batch = x.shape[0]
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, ffted.shape[-2], ffted.shape[-1])
        conv_dtype = self.conv.weight.dtype
        if ffted.dtype != conv_dtype:
            ffted = ffted.to(dtype=conv_dtype)
        ffted = self.act(self.conv(ffted))
        ffted = ffted.view(batch, -1, 2, ffted.shape[-2], ffted.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
        if ffted.dtype not in (torch.float16, torch.float32, torch.float64):
            ffted = ffted.float()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        out = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        if out.dtype != input_dtype:
            out = out.to(dtype=input_dtype)
        return out


class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fourier = FourierUnit(embed_dim // 2)
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1) if last_conv else None

    def forward(self, x):
        x_in = self.conv1(x)
        out = self.conv2(x_in + self.fourier(x_in))
        if self.last_conv is not None:
            out = self.last_conv(out)
        return out


class ResB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // red, embed_dim, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


class SFB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super().__init__()
        self.spatial = ResB(embed_dim, red=red)
        self.spectral = SpectralTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def forward(self, x):
        out = torch.cat([self.spatial(x), self.spectral(x)], dim=1)
        return self.fusion(out)


class ResidualSwinFourierBlock(nn.Module):
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
        patch_norm=True,
        resi_connection="SFB",
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
        norm_layer = nn.LayerNorm if patch_norm else None
        self.patch_embed = PatchEmbed(dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(dim)

        connection = str(resi_connection).lower()
        if connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif connection == "hsfb":
            self.conv = SFB(dim, red=2)
        elif connection == "identity":
            self.conv = nn.Identity()
        else:
            self.conv = SFB(dim)

    def forward(self, x, h, w):
        residual = x
        for blk in self.blocks:
            x = blk(x, h, w)
        x = self.patch_unembed(x, h, w)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x + residual


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


class AnimeSwin2SR(nn.Module):
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
            layer = ResidualSwinBlockV2(
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


class AnimeSwinFIR(nn.Module):
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
        patch_norm=True,
        img_range=1.0,
        upsampler="pixelshuffle",
        resi_connection="SFB",
    ):
        super().__init__()
        self.scale = scale
        self.window_size = window_size
        self.img_range = float(img_range)
        self.upsampler = str(upsampler).lower()
        if in_channels == 3:
            self.register_buffer("mean", torch.tensor((0.3014, 0.3152, 0.3094)).view(1, 3, 1, 1), persistent=False)
        else:
            self.register_buffer("mean", torch.zeros(1, in_channels, 1, 1), persistent=False)

        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        norm_layer = nn.LayerNorm if bool(patch_norm) else None
        self.patch_embed = PatchEmbed(embed_dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_idx = 0
        for i in range(len(depths)):
            layer = ResidualSwinFourierBlock(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[dpr_idx : dpr_idx + depths[i]],
                patch_norm=bool(patch_norm),
                resi_connection=resi_connection,
            )
            dpr_idx += depths[i]
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.upsample = Upsample(scale, 64)
            self.conv_last = nn.Conv2d(64, in_channels, 3, 1, 1)
        else:
            self.conv_before_upsample = None
            self.upsample = None
            self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, 1, 1)

    def forward_features(self, x):
        b, _, h, w = x.shape
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, h, w)
        x = self.norm(x)
        return self.patch_unembed(x, h, w)

    def forward(self, x):
        mean = self.mean.type_as(x)
        x = (x - mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x

        if self.upsampler == "pixelshuffle":
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
        else:
            x = self.conv_last(x)

        return x / self.img_range + mean


def _disc_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_spectral_norm=True):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    return spectral_norm(conv) if use_spectral_norm else conv


class UNetDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_spectral_norm=True, skip_connection=True):
        super().__init__()
        self.skip_connection = bool(skip_connection)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv_in = _disc_conv(in_channels, base_channels, use_spectral_norm=use_spectral_norm)
        self.down1 = _disc_conv(base_channels, base_channels * 2, 4, 2, 1, use_spectral_norm)
        self.down2 = _disc_conv(base_channels * 2, base_channels * 4, 4, 2, 1, use_spectral_norm)
        self.down3 = _disc_conv(base_channels * 4, base_channels * 8, 4, 2, 1, use_spectral_norm)

        self.up1 = _disc_conv(base_channels * 8, base_channels * 4, use_spectral_norm=use_spectral_norm)
        self.up2 = _disc_conv(base_channels * 4, base_channels * 2, use_spectral_norm=use_spectral_norm)
        self.up3 = _disc_conv(base_channels * 2, base_channels, use_spectral_norm=use_spectral_norm)

        self.conv_hr = _disc_conv(base_channels, base_channels, use_spectral_norm=use_spectral_norm)
        self.conv_out = _disc_conv(base_channels, 1, use_spectral_norm=use_spectral_norm)

    def forward(self, x):
        x0 = self.act(self.conv_in(x))
        x1 = self.act(self.down1(x0))
        x2 = self.act(self.down2(x1))
        x3 = self.act(self.down3(x2))

        x = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.up1(x))
        if self.skip_connection:
            x = x + x2

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.up2(x))
        if self.skip_connection:
            x = x + x1

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.act(self.up3(x))
        if self.skip_connection:
            x = x + x0

        x = self.act(self.conv_hr(x))
        return self.conv_out(x)


def build_model(cfg, scale):
    model_type = str(cfg.get("type", "swinir")).lower()
    common_kwargs = {
        "scale": scale,
        "in_channels": int(cfg.get("in_channels", 3)),
        "embed_dim": int(cfg.get("embed_dim", 96)),
        "depths": tuple(cfg.get("depths", (6, 6, 6, 6))),
        "num_heads": tuple(cfg.get("num_heads", (6, 6, 6, 6))),
        "window_size": int(cfg.get("window_size", 8)),
        "mlp_ratio": float(cfg.get("mlp_ratio", 4.0)),
        "qkv_bias": bool(cfg.get("qkv_bias", True)),
        "drop_rate": float(cfg.get("drop_rate", 0.0)),
        "attn_drop_rate": float(cfg.get("attn_drop_rate", 0.0)),
        "drop_path_rate": float(cfg.get("drop_path_rate", 0.1)),
    }
    if model_type == "swin2sr":
        return AnimeSwin2SR(**common_kwargs)
    if model_type == "swinfir":
        return AnimeSwinFIR(
            **common_kwargs,
            patch_norm=bool(cfg.get("patch_norm", True)),
            img_range=float(cfg.get("img_range", 1.0)),
            upsampler=str(cfg.get("upsampler", "pixelshuffle")),
            resi_connection=str(cfg.get("resi_connection", "SFB")),
        )
    return AnimeSwinIR(**common_kwargs)


def build_discriminator(cfg):
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None
    disc_type = str(cfg.get("type", "unet")).lower()
    if disc_type != "unet":
        raise ValueError(f"Unsupported discriminator type: {disc_type}")
    return UNetDiscriminator(
        in_channels=int(cfg.get("in_channels", 3)),
        base_channels=int(cfg.get("base_channels", 64)),
        use_spectral_norm=bool(cfg.get("spectral_norm", True)),
        skip_connection=bool(cfg.get("skip_connection", True)),
    )
