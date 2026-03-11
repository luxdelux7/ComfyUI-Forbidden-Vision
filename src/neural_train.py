import random
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import kornia

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

CONF = {
    "epoch": 100,
    "batch": 16,
    "lr": 8.0e-5,
    "wd": 1e-4,
    "res": 384,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 30,
    "min_epoch_to_save": 50,
    "paths": {
        "processed_train": Path(r"G:\Data\color\paired\processed_384_buckets4\train"),
        "processed_val":   Path(r"G:\Data\color\paired\processed_384_buckets4\val2"),
        "test_in":         Path(r"G:\Data\color\tests"),
        "out":             Path(r"G:\Data\color\train_mbnet_combined"),
    }
}

BASE_LOSS_WEIGHTS = {
    "rgb":                      1.0,
    "ms_ssim":                  0.4,
    "wb_reg":                   0.05,
    "wb_alignment":             5.0,
    "wb_global_cast": 3.0,
    "wb_tonal_band":            1.5,
    "luma_match":               5.0,
    "input_hi_brake":          15.0,
    "hi_grad_preserve":        10.0,
    "colored_hi_anti_desat":    2.0,
    "hue_band_chroma_v2": 4.0,
    "detail_preserve":         10.0,
    "chroma_grad":              3.0,
    "local_tonal_contrast":     12.0,
    "lower_mid_tone_push": 3.0,
    "shadow_density":           3.0,
    "shadow_reg":              0.05,
    "shadow_detail":            10.0,
    "yuv_match":                5.0,
    "uv_energy":                2.0,
    "input_neutral_preserve":   7.0,
    "target_neutral_preserve":  6.0,
    "raw_overload":            10.0,
    "lightroom_rails":         15.0,
    "tv_grid":               0.0003,
    "tv_chroma_grid":        0.001,
    "global_reg":             0.001,
}

_M_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

_M_XYZ2RGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

_M_BRADFORD = torch.tensor([
    [ 0.8951,  0.2664, -0.1614],
    [-0.7502,  1.7135,  0.0367],
    [ 0.0389, -0.0685,  1.0296],
], dtype=torch.float32)

_M_BRADFORD_INV = torch.tensor([
    [ 0.9869929, -0.1470543,  0.1599627],
    [ 0.4323053,  0.5183603,  0.0492912],
    [-0.0085287,  0.0400428,  0.9684867],
], dtype=torch.float32)

def _ensure_mat_device(mat, like):
    return mat.to(device=like.device, dtype=like.dtype)

def _xy_to_XYZ_white(xy):
    x = xy[:, 0].clamp(1e-6, 1.0)
    y = xy[:, 1].clamp(1e-6, 1.0)
    Y = torch.ones_like(x)
    X = x * (Y / y)
    Z = (1.0 - x - y) * (Y / y)
    return torch.stack([X, Y, Z], dim=1)

def _cct_to_xy_approx(cct):
    T = cct.clamp(1667.0, 25000.0)
    x = torch.where(
        T <= 4000.0,
        (-0.2661239e9 / (T**3)) - (0.2343580e6 / (T**2)) + (0.8776956e3 / T) + 0.179910,
        (-3.0258469e9 / (T**3)) + (2.1070379e6 / (T**2)) + (0.2226347e3 / T) + 0.240390,
    )
    y = torch.where(
        T <= 2222.0,
        -1.1063814*(x**3) - 1.34811020*(x**2) + 2.18555832*x - 0.20219683,
        torch.where(
            T <= 4000.0,
            -0.9549476*(x**3) - 1.37418593*(x**2) + 2.09137015*x - 0.16748867,
            3.0817580*(x**3) - 5.87338670*(x**2) + 3.75112997*x - 0.37001483
        )
    )
    return torch.stack([x.clamp(0.0, 0.99), y.clamp(0.0, 0.99)], dim=1)

def _xy_to_uv1960(xy):
    x, y = xy[:, 0], xy[:, 1]
    denom = (-2.0*x + 12.0*y + 3.0).clamp_min(1e-6)
    return torch.stack([(4.0*x)/denom, (6.0*y)/denom], dim=1)

def _uv1960_to_xy(uv):
    u, v = uv[:, 0], uv[:, 1]
    denom = (2.0*u - 8.0*v + 4.0).clamp_min(1e-6)
    x = (3.0*u) / denom
    y = (2.0*v) / denom
    return torch.stack([x.clamp(0.0, 0.99), y.clamp(0.0, 0.99)], dim=1)

def _apply_3x3_to_image(mat3, img3):
    B, C, H, W = img3.shape
    x = img3.permute(0, 2, 3, 1).reshape(B, -1, 3)
    if mat3.dim() == 2:
        m = mat3.unsqueeze(0).expand(B, -1, -1)
    else:
        m = mat3
    y = torch.bmm(x, m.transpose(1, 2))
    return y.reshape(B, H, W, 3).permute(0, 3, 1, 2)

def apply_wb_temp_tint_bradford(rgb_srgb, temp, tint, *,
                                 base_cct=6500.0, temp_mired_range=120.0,
                                 tint_uv_range=0.06, preserve_luma=True, eps=1e-6):
    B = rgb_srgb.shape[0]
    t = temp.view(B).clamp(-1.0, 1.0)
    k = tint.view(B).clamp(-1.0, 1.0)

    lin = srgb_to_linear(rgb_srgb)
    if preserve_luma:
        orig_luma = 0.2126 * lin[:, 0:1] + 0.7152 * lin[:, 1:2] + 0.0722 * lin[:, 2:3]

    base_mired = 1e6 / base_cct
    delta_mired = t * temp_mired_range
    cct = (1e6 / (base_mired + delta_mired).clamp_min(1.0)).clamp(2000.0, 20000.0)

    xy = _cct_to_xy_approx(cct)
    uv = _xy_to_uv1960(xy)
    uv = torch.stack([uv[:, 0], uv[:, 1] + k * tint_uv_range], dim=1)
    xy_tinted = _uv1960_to_xy(uv)

    src_xy = xy_tinted.new_tensor([0.3127, 0.3290]).unsqueeze(0).expand(B, 2)
    src_XYZ = _xy_to_XYZ_white(src_xy)
    dst_XYZ = _xy_to_XYZ_white(xy_tinted)

    M_b = _ensure_mat_device(_M_BRADFORD, lin)
    M_bi = _ensure_mat_device(_M_BRADFORD_INV, lin)
    M_rgb2xyz = _ensure_mat_device(_M_RGB2XYZ, lin)
    M_xyz2rgb = _ensure_mat_device(_M_XYZ2RGB, lin)

    src_LMS = torch.matmul(src_XYZ, M_b.T)
    dst_LMS = torch.matmul(dst_XYZ, M_b.T)
    D = (dst_LMS / (src_LMS + eps)).clamp(0.25, 4.0)

    Dm = torch.zeros(B, 3, 3, device=lin.device, dtype=lin.dtype)
    Dm[:, 0, 0] = D[:, 0]; Dm[:, 1, 1] = D[:, 1]; Dm[:, 2, 2] = D[:, 2]
    A = torch.bmm(torch.bmm(M_bi.unsqueeze(0).expand(B,-1,-1), Dm),
                   M_b.unsqueeze(0).expand(B,-1,-1))

    xyz = _apply_3x3_to_image(M_rgb2xyz, lin)
    xyz_adapted = _apply_3x3_to_image(A, xyz)
    lin_out = _apply_3x3_to_image(M_xyz2rgb, xyz_adapted)

    if preserve_luma:
        new_luma = 0.2126 * lin_out[:, 0:1] + 0.7152 * lin_out[:, 1:2] + 0.0722 * lin_out[:, 2:3]
        lin_out = lin_out * (orig_luma / (new_luma + eps))

    return linear_to_srgb(lin_out.clamp(0.0, 1.0)).clamp(0.0, 1.0)
def _uv_from_rgb_bt601(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    u = -0.14713 * r - 0.28886 * g + 0.436  * b
    v =  0.615   * r - 0.51499 * g - 0.10001* b
    return u, v

def rgb_to_yuv_bt601(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436  * b
    v =  0.615   * r - 0.51499 * g - 0.10001* b
    return torch.cat([y, u, v], dim=1)

def yuv_to_rgb_bt601(yuv):
    y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.cat([r, g, b], dim=1)

def _luma601(x):
    return 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]

def _apply_saturation_chw(x, sat):
    y = _luma601(x)
    return y + (x - y) * sat

def _apply_contrast_chw(x, c):
    y  = _luma601(x)
    mu = y.mean(dim=(1, 2), keepdim=True)
    return mu + (x - mu) * c

def srgb_to_linear(x):
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    x = x.clamp(0.0, 1.0)
    x_safe = x.clamp_min(1e-5)
    srgb_curve = 1.055 * (x_safe ** (1.0 / 2.4)) - 0.055
    return torch.where(x <= 0.0031308, x * 12.92, srgb_curve)

def apply_wb_gains_linear(rgb_srgb, r_gain, b_gain):
    linear = srgb_to_linear(rgb_srgb)
    
    r = linear[:, 0:1] * r_gain
    g = linear[:, 1:2]
    b = linear[:, 2:3] * b_gain
    
    orig_luma = 0.2126 * linear[:, 0:1] + 0.7152 * linear[:, 1:2] + 0.0722 * linear[:, 2:3]
    new_luma  = 0.2126 * r        + 0.7152 * g        + 0.0722 * b
    
    ratio = orig_luma / (new_luma + 1e-6)
    
    r_out = r * ratio
    g_out = g * ratio
    b_out = b * ratio
    
    return linear_to_srgb(torch.cat([r_out, g_out, b_out], dim=1).clamp(0.0, 1.0))

def soft_rolloff(x, knee=0.85, limit=1.05):
    x = F.relu(x)
    scale = limit - knee
    return torch.where(x <= knee, x, limit - scale * torch.exp(-(x - knee) / scale))

def dynamic_highlight_recovery(y_pred, rgb_pred, strength, threshold=0.96):
    max_rgb = rgb_pred.amax(dim=1, keepdim=True)
    gate    = ((max_rgb - threshold) / (1.0 - threshold + 1e-6)).clamp(0.0, 1.0)
    gate    = gate * gate * (3.0 - 2.0 * gate)
    return y_pred * (1.0 - gate * strength * 0.30)

def restore_highlight_detail_pointwise(yuv, luma_pre, lo=0.72, hi=0.98, max_restore=0.03):
    y_current = yuv[:, 0:1]
    t    = ((luma_pre - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)
    diff    = luma_pre - y_current
    restore = diff.clamp(-max_restore, max_restore) * mask
    y_new   = (y_current + restore).clamp(0.004, 0.996)
    return torch.cat([y_new, yuv[:, 1:2], yuv[:, 2:3]], dim=1)

def tone_aug_x_only(x, x_target=None):
    """
    Safe input-only augmentation for paired color/tone training.

    Goals:
    - improve robustness to unseen SDXL / anime / photo-like inputs
    - avoid redefining the target look
    - avoid making every image look like it needs strong correction
    - stay conservative so the model doesn't learn to over-brighten everything
    """
    x = x.clamp(0.0, 1.0)

    if torch.rand(1).item() >= 0.55:
        return x

    if torch.rand(1).item() < 0.80:
        if torch.rand(1).item() < 0.75:
            ev = torch.empty(1).uniform_(-0.18, 0.18).item()
        else:
            ev = torch.empty(1).uniform_(-0.30, 0.30).item()
        x = x * (2.0 ** ev)

    if torch.rand(1).item() < 0.60:
        gamma = torch.empty(1).uniform_(0.93, 1.08).item()
        x = x.clamp(0.0, 1.0).pow(gamma)

    if torch.rand(1).item() < 0.55:
        c = torch.empty(1).uniform_(0.92, 1.08).item()
        x = _apply_contrast_chw(x, float(c))

    if torch.rand(1).item() < 0.40:
        s = torch.empty(1).uniform_(0.94, 1.06).item()
        x = _apply_saturation_chw(x, float(s))

    if torch.rand(1).item() < 0.18:
        B = 1
        temp = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(-0.10, 0.10)
        tint = torch.empty(B, 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(-0.05, 0.05)
        x = apply_wb_temp_tint_bradford(
            x.unsqueeze(0),
            temp=temp,
            tint=tint,
            base_cct=6500.0,
            temp_mired_range=80.0,
            tint_uv_range=0.035,
            preserve_luma=True,
        ).squeeze(0)

    if torch.rand(1).item() < 0.18:
        sigma = torch.empty(1).uniform_(0.35, 0.90).item()
        k = int(max(3, round(sigma * 6))) | 1
        x = kornia.filters.gaussian_blur2d(
            x.unsqueeze(0), (k, k), (sigma, sigma)
        ).squeeze(0)

    if torch.rand(1).item() < 0.20:
        noise_std = torch.empty(1).uniform_(0.002, 0.008).item()
        x = x + torch.randn_like(x) * noise_std

    return x.clamp(0.0, 1.0)
def paired_crop_scale_jitter(x, y, p=0.35, min_scale=0.90):
    """
    Safe paired crop jitter.
    Apply to BOTH input and target identically.
    Keeps global edit relationship intact while reducing composition overfit.
    """
    if torch.rand(1).item() >= p:
        return x, y

    _, H, W = x.shape
    scale = torch.empty(1).uniform_(min_scale, 1.0).item()

    new_h = max(8, int(round(H * scale)))
    new_w = max(8, int(round(W * scale)))

    if new_h >= H or new_w >= W:
        return x, y

    top = torch.randint(0, H - new_h + 1, (1,)).item()
    left = torch.randint(0, W - new_w + 1, (1,)).item()

    x_crop = x[:, top:top + new_h, left:left + new_w].unsqueeze(0)
    y_crop = y[:, top:top + new_h, left:left + new_w].unsqueeze(0)

    x = F.interpolate(x_crop, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
    y = F.interpolate(y_crop, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

    return x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)

@torch.no_grad()
def image_stats_10(x):
    eps = 1e-6
    B, _, H, W = x.shape
    x  = x.clamp(0.0, 1.0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    l  = 0.299 * r + 0.587 * g + 0.114 * b
    lf = l.view(B, -1)
    mean_l = lf.mean(dim=1)
    std_l  = lf.std(dim=1).clamp(0.0, 0.5) * 2.0
    q_levels = torch.tensor([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99], device=x.device)
    q  = torch.quantile(lf, q_levels, dim=1).transpose(0, 1)
    bins = 16
    idx  = torch.clamp((lf * (bins - 1)).long(), 0, bins - 1)
    hist = torch.zeros(B, bins, device=x.device, dtype=x.dtype)
    hist.scatter_add_(1, idx, torch.ones_like(lf))
    hist = hist / (lf.shape[1] + eps)
    u      = -0.14713*r - 0.28886*g + 0.436*b
    v      =  0.61500*r - 0.51499*g - 0.10001*b
    chroma = torch.sqrt(u*u + v*v + 1e-8)
    m_sh   = (l < 0.25).float()
    m_md   = ((l >= 0.25) & (l < 0.75)).float()
    m_hi   = (l >= 0.75).float()
    def band_mean(t, m):
        return (t * m).sum(dim=(1,2)) / m.sum(dim=(1,2)).clamp_min(1.0)
    chroma_sh = band_mean(chroma, m_sh)
    chroma_md = band_mean(chroma, m_md)
    chroma_hi = band_mean(chroma, m_hi)
    pct_sh, pct_md, pct_hi = m_sh.mean(dim=(1,2)), m_md.mean(dim=(1,2)), m_hi.mean(dim=(1,2))
    hi_r = F.relu(r - 0.98).mean(dim=(1,2));  hi_g = F.relu(g - 0.98).mean(dim=(1,2))
    hi_b = F.relu(b - 0.98).mean(dim=(1,2))
    lo_r = F.relu(0.02-r).mean(dim=(1,2));    lo_g = F.relu(0.02-g).mean(dim=(1,2))
    lo_b = F.relu(0.02-b).mean(dim=(1,2))
    r_mean = r.mean(dim=(1,2)).clamp_min(eps); g_mean = g.mean(dim=(1,2)).clamp_min(eps)
    b_mean = b.mean(dim=(1,2)).clamp_min(eps)
    log_rg = torch.log(r_mean / g_mean);      log_bg = torch.log(b_mean / g_mean)
    u_std  = u.view(B,-1).std(dim=1).clamp(0.0,0.25)*4.0
    v_std  = v.view(B,-1).std(dim=1).clamp(0.0,0.25)*4.0
    ch0,ch1 = H//4, H-(H//4);  cw0,cw1 = W//4, W-(W//4)
    lcf    = l[:,ch0:ch1,cw0:cw1].contiguous().view(B,-1)
    c_mean = lcf.mean(dim=1);  c_std = lcf.std(dim=1).clamp(0.0,0.5)*2.0
    c_q    = torch.quantile(lcf, torch.tensor([0.10,0.50,0.90],device=x.device), dim=1).transpose(0,1)
    c_clip = ((x[:,:,ch0:ch1,cw0:cw1]>0.995).any(dim=1)).float().mean(dim=(1,2))
    hue    = torch.atan2(v, u)
    hue_norm = (hue/(2.0*3.14159265359)+0.5)%1.0
    hue_bin_idx = (hue_norm*12.0).long().clamp(0,11).view(B,-1)
    sat_flat    = chroma.view(B,-1)
    hue_hist    = torch.zeros(B,12,device=x.device,dtype=x.dtype)
    hue_hist.scatter_add_(1, hue_bin_idx, sat_flat)
    hue_hist = hue_hist/(H*W+eps)*10.0
    return torch.cat([
        mean_l[:,None], std_l[:,None], q, hist,
        chroma_sh[:,None], chroma_md[:,None], chroma_hi[:,None],
        pct_sh[:,None], pct_md[:,None], pct_hi[:,None],
        hi_r[:,None], hi_g[:,None], hi_b[:,None],
        lo_r[:,None], lo_g[:,None], lo_b[:,None],
        log_rg[:,None], log_bg[:,None], u_std[:,None], v_std[:,None],
        c_mean[:,None], c_std[:,None], c_q, c_clip[:,None], hue_hist
    ], dim=1)

@torch.no_grad()
def correction_need_score(x_in, y_tgt):
    lx = 0.299*x_in[:,0]+0.587*x_in[:,1]+0.114*x_in[:,2]
    ly = 0.299*y_tgt[:,0]+0.587*y_tgt[:,1]+0.114*y_tgt[:,2]
    m_mid = ((ly-0.10)/0.40).clamp(0,1)*((0.90-ly)/0.30).clamp(0,1)
    m_mid = (m_mid/m_mid.max(dim=2,keepdim=True)[0].max(dim=1,keepdim=True)[0].clamp_min(1e-6)).clamp(0,1)
    d_mid = ((ly-lx).abs()*m_mid).sum(dim=(1,2))/(m_mid.sum(dim=(1,2))+1e-6)
    m_hi  = (((ly-0.80)/0.18).clamp(0,1))
    m_hi  = m_hi*m_hi*(3.0-2.0*m_hi)
    d_hi  = ((ly-lx).abs()*m_hi).sum(dim=(1,2))/(m_hi.sum(dim=(1,2))+1e-6)
    ux,vx = _uv_from_rgb_bt601(x_in);  uy,vy = _uv_from_rgb_bt601(y_tgt)
    d_uv  = (uy.mean(dim=(1,2))-ux.mean(dim=(1,2))).abs()+(vy.mean(dim=(1,2))-vx.mean(dim=(1,2))).abs()
    d_chroma = (torch.sqrt(uy**2+vy**2+1e-8)-torch.sqrt(ux**2+vx**2+1e-8)).abs().mean(dim=(1,2))
    return (d_mid*10.0+d_hi*6.0+d_uv*6.0+d_chroma*8.0).clamp(0.0,1.0)


def _smoothstep(t):
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _get_luma(x):
    return 0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]
def lower_mid_tone_push_loss(
    pred,
    tgt,
    x_in,
    lo=0.14,
    hi=0.55,
    under_weight=1.20,
    over_weight=0.40,
    detail_ref_mix=0.5,
):
    """
    Target-guided lower-mid luma matching.

    Purpose:
    - push underlit lower mids upward toward target
    - allow a little overshoot instead of staying timid
    - keep behavior focused away from deep blacks / highlights
    """
    eps = 1e-6

    lp = _get_luma(pred)
    lt = _get_luma(tgt)
    li = _get_luma(x_in)

    gate = _smoothstep((lt - lo) / (0.08 + eps))
    gate = gate * (1.0 - _smoothstep((lt - hi) / (0.10 + eps)))

    need = (lt - li).abs()
    need_gate = (0.75 + 1.25 * _smoothstep((need - 0.01) / (0.06 + eps))).detach()

    m = (gate * need_gate).detach()

    if m.sum() < 1.0:
        return pred.new_tensor(0.0)

    under = F.relu(lt - lp)
    over = F.relu(lp - lt)

    def local_amp(x, s1=2.0, s2=6.0):
        k1 = int(max(3, round(s1 * 6))) | 1
        k2 = int(max(3, round(s2 * 6))) | 1
        b1 = kornia.filters.gaussian_blur2d(x, (k1, k1), (s1, s1))
        b2 = kornia.filters.gaussian_blur2d(x, (k2, k2), (s2, s2))
        return (b1 - b2).abs()

    amp_t = local_amp(lt)
    amp_i = local_amp(li)
    amp_ref = torch.maximum(amp_t, amp_i * detail_ref_mix)
    amp_p = local_amp(lp)

    detail_loss = F.relu(amp_ref - amp_p)

    loss = (
        (under_weight * under + over_weight * over) * m
    ).sum() / (m.sum() + eps)

    detail_term = (detail_loss * m).sum() / (m.sum() + eps)

    return loss + 0.20 * detail_term
def toward_target_loss(
    pred,
    tgt,
    x_in,
    x_orig=None,
    mask=None,
    *,
    overshoot_weight=0.35,
    regression_weight=1.15,
    completion_weight=0.18,
    overshoot_scale=None,
    eps=1e-6,
):
    """
    Safer rework of the original toward-target loss.

    Compared to the original:
    - keeps the stable baseline-relative structure
    - still penalizes getting worse than baseline
    - slightly reduces pure overshoot fear
    - adds a small completion push in harder regions
    - remains much more optimizer-friendly than a full rewrite
    """
    if x_orig is not None:
        err_in   = (x_in   - tgt).abs()
        err_orig = (x_orig - tgt).abs()
        temp = 8.0
        w_orig = torch.softmax(torch.stack([-err_orig * temp, -err_in * temp], dim=0), dim=0)[0]
        baseline = w_orig * x_orig + (1.0 - w_orig) * x_in
        err_base = w_orig * err_orig + (1.0 - w_orig) * err_in
    else:
        baseline = x_in
        err_base = (x_in - tgt).abs()

    err_pred = (pred - tgt).abs()

    regression = F.relu(err_pred - err_base)

    sign_base = (baseline - tgt).sign()
    sign_pred = (pred - tgt).sign()
    crossed = (sign_base * sign_pred < 0).float()

    overshoot = err_pred * crossed
    if overshoot_scale is not None:
        overshoot = overshoot * overshoot_scale

    hard_region = (err_base / (err_base.mean(dim=(1, 2, 3), keepdim=True) + eps)).clamp(0.5, 2.5)
    completion = err_pred * hard_region

    loss_map = (
        regression_weight * regression +
        overshoot_weight * overshoot +
        err_pred * 0.10 +
        completion_weight * completion
    )

    if mask is not None:
        mask = mask.detach()
        return (loss_map * mask).sum() / (mask.sum() + eps)

    return loss_map.mean()

class Charbonnier(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__(); self.eps = eps
    def forward(self, pred, tgt):
        d = pred - tgt
        return torch.mean(torch.sqrt(d*d + self.eps*self.eps))

def multiscale_luma_charbonnier_loss(pred, tgt, eps=1e-3):
    lp = _get_luma(pred);  lt = _get_luma(tgt)
    def charb(a, b):
        d = a - b
        return torch.mean(torch.sqrt(d*d + eps*eps))
    total = lp.new_tensor(0.0)
    for f, s, w in zip((1,2,4),(0.0,0.8,1.6),(0.50,0.30,0.20)):
        a, b = lp, lt
        if s > 0:
            k = int(max(3, round(s*6)))|1
            a = kornia.filters.gaussian_blur2d(a,(k,k),(s,s))
            b = kornia.filters.gaussian_blur2d(b,(k,k),(s,s))
        if f > 1:
            a = F.avg_pool2d(a,f,f);  b = F.avg_pool2d(b,f,f)
        total = total + charb(a,b)*w
    return total

def wb_gain_regularization_loss(r_gain, b_gain, weight=1.0):
    """Gentle pull toward 1.0, just enough to prevent wild drifts."""
    return (((r_gain - 1.0)**2).mean() +
            ((b_gain - 1.0)**2).mean()) * weight

def highlight_gradient_preserve_loss(pred, x_orig, tgt=None, lo=0.80, hi=1.00,
                                      target_weight=0.5, weight=1.0):
    eps = 1e-6
    lp  = _get_luma(pred);   lx = _get_luma(x_orig)
    t   = ((lx - lo)/(hi - lo + eps)).clamp(0.0,1.0)
    mask = (t*t*(3.0-2.0*t)).detach()
    def grads(img):
        gh = F.pad(img[:,:,:,1:]-img[:,:,:,:-1],(0,1,0,0))
        gv = F.pad(img[:,:,1:,:]-img[:,:,:-1,:],(0,0,0,1))
        return torch.sqrt(gh*gh+gv*gv+eps)
    if mask.sum() < 1.0:
        return pred.new_tensor(0.0)
    loss = (F.relu(grads(lx)-grads(lp))*mask).sum()/(mask.sum()+eps)
    if tgt is not None:
        lt   = _get_luma(tgt)
        loss = loss+(F.relu(grads(lt)-grads(lp))*mask).sum()/(mask.sum()+eps)*target_weight
    return loss*weight

def input_highlight_brake_loss(pred, x_in, lo=0.88, weight=1.0):
    lp   = _get_luma(pred);  lx = _get_luma(x_in)
    t    = ((lx-lo)/(1.0-lo+1e-6)).clamp(0.0,1.0)
    mask = t*t*(3.0-2.0*t)
    overshoot = F.relu(lp-(lx+0.005))
    return (overshoot*mask).sum()/(mask.sum()+1e-6)*weight

def full_range_detail_preserve_loss(pred, tgt, x_in, luma_lo=0.05, luma_hi=0.99, weight=1.0):
    eps = 1e-6
    lp,lt,lx = _get_luma(pred),_get_luma(tgt),_get_luma(x_in)
    def make_gate(l):
        t_lo = _smoothstep((l-luma_lo)/(0.05+eps))
        t_hi = _smoothstep((l-luma_hi)/(0.04+eps))
        return t_lo*(1.0-t_hi)
    mask  = torch.maximum(make_gate(lt),make_gate(lx)).detach()
    total = pred.new_tensor(0.0)
    for (s1,s2,md),sw in zip([(0.8,2.2,0.004),(2.0,5.0,0.003)],[0.6,0.4]):
        k1 = int(max(3,round(s1*6)))|1;  k2 = int(max(3,round(s2*6)))|1
        det_p = (kornia.filters.gaussian_blur2d(lp,(k1,k1),(s1,s1))-
                 kornia.filters.gaussian_blur2d(lp,(k2,k2),(s2,s2))).abs()
        det_t = (kornia.filters.gaussian_blur2d(lt,(k1,k1),(s1,s1))-
                 kornia.filters.gaussian_blur2d(lt,(k2,k2),(s2,s2))).abs()
        det_x = (kornia.filters.gaussian_blur2d(lx,(k1,k1),(s1,s1))-
                 kornia.filters.gaussian_blur2d(lx,(k2,k2),(s2,s2))).abs()
        det_ref = torch.maximum(det_t,det_x)
        m = (mask*(det_ref>md).float()).detach()
        if m.sum()<1.0: continue
        total = total+(F.relu(det_ref-det_p)*m).sum()/(m.sum()+eps)*sw
    return total*weight

def shadow_density_loss(pred, tgt, lo=0.03, hi=0.20, weight=1.0):
    eps = 1e-6
    lp,lt = _get_luma(pred),_get_luma(tgt)
    t     = ((lt-lo)/(hi-lo+eps)).clamp(0.0,1.0)
    m_sh  = 1.0-t*t*(3.0-2.0*t)
    lift  = F.relu(lp - lt)
    crush = F.relu(lt - lp)
    base  = ((lift + 0.7 * crush) * m_sh).sum() / (m_sh.sum() + eps)
    b1t   = kornia.filters.gaussian_blur2d(lt,(7,7),(0.8,0.8))
    b2t   = kornia.filters.gaussian_blur2d(lt,(13,13),(2.2,2.2))
    m_tex = ((b1t-b2t).abs()>0.010).float()
    m     = (m_sh*m_tex).detach()
    undercut  = F.relu((lt-lp)-0.004)
    tex_guard = (undercut*m).sum()/(m.sum()+eps) if m.sum()>1.0 else lp.new_tensor(0.0)
    return (base+0.20*tex_guard)*weight

def hue_band_chroma_match_loss_v2(
    pred,
    tgt,
    luma_lo=0.03,
    luma_hi=0.97,
    chroma_lo=0.018,
    chroma_hi=0.28,
    num_hues=8,
    under_weight=1.0,
    over_weight=0.75,
    hue_weight=0.16,
    downsample=2,
    weight=1.0,
):
    """
    Faster target-guided smooth HSL-like chroma matching.

    Speedups vs previous v2:
    - computes on downsampled maps
    - uses fewer hue centers
    - hue-direction term only on rich bins
    - precomputes shared maps once

    Still:
    - matches target chroma both upward and downward
    - separates hue/chroma/luma more than the old broad hue-band loss
    - stays fully smooth/differentiable
    """
    eps = 1e-6
    pi = 3.14159265359

    def ss(x, e0, e1):
        t = ((x - e0) / (e1 - e0 + eps)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def get_stats(x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v =  0.615   * r - 0.51499 * g - 0.10001 * b
        return y, u, v

    yp, up, vp = get_stats(pred)
    yt, ut, vt = get_stats(tgt)

    if downsample > 1:
        yp = F.avg_pool2d(yp, downsample, downsample)
        up = F.avg_pool2d(up, downsample, downsample)
        vp = F.avg_pool2d(vp, downsample, downsample)

        yt = F.avg_pool2d(yt, downsample, downsample)
        ut = F.avg_pool2d(ut, downsample, downsample)
        vt = F.avg_pool2d(vt, downsample, downsample)

    cp = torch.sqrt(up * up + vp * vp + eps)
    ct = torch.sqrt(ut * ut + vt * vt + eps)

    hp = torch.atan2(vp, up)
    ht = torch.atan2(vt, ut)

    valid_luma = ss(yt, luma_lo, luma_lo + 0.06) * (1.0 - ss(yt, luma_hi - 0.03, luma_hi))
    valid_chroma = ss(ct, chroma_lo, chroma_lo + 0.05)
    base_gate = (valid_luma * valid_chroma).detach()

    if base_gate.sum() < 1.0:
        return pred.new_tensor(0.0)

    under = F.relu(ct - cp)
    over  = F.relu(cp - ct)
    strength = (1.0 + 1.1 * ss(ct, 0.05, 0.16)).detach()

    uv_pred = torch.cat([up, vp], dim=1)
    uv_tgt  = torch.cat([ut, vt], dim=1)
    cos_sim = F.cosine_similarity(uv_pred, uv_tgt, dim=1, eps=1e-8).unsqueeze(1)

    centers = torch.linspace(-pi, pi, steps=num_hues + 1, device=pred.device, dtype=pred.dtype)[:-1]
    band_width = (2.0 * pi) / float(num_hues)
    sigma = band_width * 0.42

    chroma_bins = [
        (0.02, 0.10, 0.90, False),
        (0.10, 0.30, 1.20, True),
    ]
    luma_bins = [
        (0.05, 0.55, 1.10),
        (0.55, 0.90, 0.90),
    ]

    total_chroma = pred.new_tensor(0.0)
    total_hue = pred.new_tensor(0.0)
    total_weight = pred.new_tensor(0.0)

    for center in centers:
        hue_dist = torch.atan2(torch.sin(ht - center), torch.cos(ht - center)).abs()
        hue_gate = torch.exp(-(hue_dist * hue_dist) / (2.0 * sigma * sigma))

        hue_base = base_gate * hue_gate

        for c0, c1, cw, use_hue_term in chroma_bins:
            chroma_gate = ss(ct, c0, min(c0 + 0.04, c1)) * (1.0 - ss(ct, max(c1 - 0.04, c0), c1))

            for y0, y1, yw in luma_bins:
                luma_gate = ss(yt, y0, min(y0 + 0.08, y1)) * (1.0 - ss(yt, max(y1 - 0.08, y0), y1))

                m = (hue_base * chroma_gate * luma_gate).detach()
                denom = m.sum()

                if denom < 24.0:
                    continue

                bin_weight = cw * yw

                chroma_bin = (
                    ((under_weight * under) + (over_weight * over)) * strength * m
                ).sum() / (denom + eps)

                total_chroma = total_chroma + chroma_bin * bin_weight
                total_weight = total_weight + bin_weight

                if use_hue_term:
                    hue_gate2 = (m * ss(ct, 0.035, 0.10)).detach()
                    denom_h = hue_gate2.sum()
                    if denom_h > 1.0:
                        hue_bin = ((1.0 - cos_sim) * hue_gate2).sum() / (denom_h + eps)
                        total_hue = total_hue + hue_bin * bin_weight

    if total_weight.item() <= 0.0:
        return pred.new_tensor(0.0)

    total_chroma = total_chroma / total_weight
    total_hue = total_hue / total_weight

    return (total_chroma + hue_weight * total_hue) * weight
def full_range_uv_match_loss(pred, tgt, x_in, x_orig=None,
                              luma_lo=0.03, luma_hi=0.96,
                              direction_weight=0.3, overshoot_weight=0.5, weight=1.0):
    eps = 1e-6
    def get_uv(x):
        r,g,b = x[:,0],x[:,1],x[:,2]
        return -0.14713*r-0.28886*g+0.436*b, 0.615*r-0.51499*g-0.10001*b
    lt       = 0.299*tgt[:,0]+0.587*tgt[:,1]+0.114*tgt[:,2]
    up,vp    = get_uv(pred)
    ut,vt    = get_uv(tgt)
    x_ref    = x_orig if x_orig is not None else x_in
    ui,vi    = get_uv(x_ref)

    t_lo     = _smoothstep((lt-luma_lo)/(0.05+eps))
    t_hi     = _smoothstep((lt-luma_hi)/(0.03+eps))
    luma_gate = (t_lo*(1.0-t_hi))

    chroma_t      = torch.sqrt(ut*ut + vt*vt + eps)
    chroma_weight = (1.0 + 2.5 * torch.exp(-chroma_t * 35.0)).clamp(1.0, 3.5)

    gate = (luma_gate * chroma_weight).detach()

    l1_loss  = ((torch.abs(up-ut)+torch.abs(vp-vt))*gate).sum()/(gate.sum()+eps)

    crossed_u = ((ui-ut).sign()*(up-ut).sign()<0).float()
    crossed_v = ((vi-vt).sign()*(vp-vt).sign()<0).float()
    overshoot_chroma_scale = (1.0 + chroma_t * 15.0).clamp(1.0, 3.0)
    overshoot = (torch.abs(up - ut) * crossed_u + torch.abs(vp - vt) * crossed_v) * overshoot_chroma_scale
    overshoot_loss = (overshoot*gate).sum()/(gate.sum()+eps)

    chroma_gate = _smoothstep((chroma_t-0.03)/(0.05+eps))
    uv_pred   = torch.stack([up,vp],dim=1)
    uv_tgt    = torch.stack([ut,vt],dim=1)
    cos_sim   = F.cosine_similarity(uv_pred,uv_tgt,dim=1,eps=1e-8)
    dir_gate  = (luma_gate * chroma_gate * chroma_weight).detach()
    dir_denom = dir_gate.sum()
    dir_loss  = ((1.0-cos_sim)*dir_gate).sum()/(dir_denom+eps) if dir_denom>1.0 \
                else pred.new_tensor(0.0)

    return (l1_loss+overshoot_weight*overshoot_loss+direction_weight*dir_loss)*weight

def uv_energy_loss(pred, tgt, weight=1.0):
    """Symmetric chroma energy balance — penalises both dull and neon."""
    def get_uv(x):
        r,g,b = x[:,0],x[:,1],x[:,2]
        return -0.14713*r-0.28886*g+0.436*b, 0.615*r-0.51499*g-0.10001*b
    up,vp = get_uv(pred);  ut,vt = get_uv(tgt)
    diff_e = (up**2+vp**2).mean((1,2))-(ut**2+vt**2).mean((1,2))
    return (F.relu(-diff_e).pow(2)*5.0+F.relu(diff_e).pow(2)*4.0).mean()*weight
def target_neutral_preserve_loss(pred, tgt, 
                                  chroma_abs_hi=0.025,
                                  luma_lo=0.05, luma_hi=0.97,
                                  weight=1.0):
    eps = 1e-6
    def ss(x, e0, e1):
        t = ((x - e0) / (e1 - e0 + 1e-8)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    rt, gt, bt = tgt[:, 0:1], tgt[:, 1:2], tgt[:, 2:3]
    yt = 0.299 * rt + 0.587 * gt + 0.114 * bt
    ut = -0.14713 * rt - 0.28886 * gt + 0.436 * bt
    vt =  0.615   * rt - 0.51499 * gt - 0.10001 * bt
    ct = torch.sqrt(ut * ut + vt * vt + eps)

    rp, gp, bp = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    up = -0.14713 * rp - 0.28886 * gp + 0.436 * bp
    vp =  0.615   * rp - 0.51499 * gp - 0.10001 * bp
    cp = torch.sqrt(up * up + vp * vp + eps)

    neutral_t  = (1.0 - ss(ct, 0.005, chroma_abs_hi))
    luma_gate  = ss(yt, luma_lo - 0.02, luma_lo + 0.02) * \
                 (1.0 - ss(yt, luma_hi - 0.002, luma_hi + 0.002))
    m = (neutral_t * luma_gate).detach()

    if m.sum() < 1.0:
        return pred.new_tensor(0.0)

    loss = (cp * cp * m).sum() / (m.sum() + eps)
    return loss * weight
def input_neutral_preserve_loss(pred, x_in, tgt,
                                 chroma_abs_hi=0.022, chroma_ratio_hi=0.10,
                                 luma_lo=0.01, luma_hi=0.999,
                                 soft_allow_margin=0.004, soft_weight=0.35, weight=1.0):
    eps = 1e-6
    def ss(x,e0,e1):
        t = ((x-e0)/(e1-e0+1e-8)).clamp(0.0,1.0)
        return t*t*(3.0-2.0*t)
    def get_stats(x):
        r,g,b = x[:,0:1],x[:,1:2],x[:,2:3]
        y = 0.299*r+0.587*g+0.114*b
        u = -0.14713*r-0.28886*g+0.436*b
        v =  0.615*r-0.51499*g-0.10001*b
        return u,v,y,torch.sqrt(u*u+v*v+1e-8)
    ui,vi,yi,ci = get_stats(x_in)
    ut_,vt_,yt_,ct_ = get_stats(tgt)
    up_,vp_,_,cp_ = get_stats(pred)
    neutral_in = (1.0-ss(ci,0.010,chroma_abs_hi))*(1.0-ss(ci/(yi+0.02),0.04,chroma_ratio_hi))
    neutral_t  = (1.0-ss(ct_,0.010,chroma_abs_hi))*(1.0-ss(ct_/(yt_+0.02),0.04,chroma_ratio_hi))
    luma_gate  = ss(yi,luma_lo-0.02,luma_lo+0.02)*(1.0-ss(yi,luma_hi-0.002,luma_hi+0.002))
    m_strict   = (neutral_in*neutral_t*luma_gate).detach()
    m_soft     = (neutral_in*(1.0-neutral_t)*luma_gate).detach()
    loss       = pred.new_tensor(0.0)
    if m_strict.sum()>=1.0:
        loss = loss+((up_*up_+vp_*vp_)*m_strict).sum()/(m_strict.sum()+eps)
    if m_soft.sum()>=1.0:
        loss = loss+(F.relu(cp_-(ct_+soft_allow_margin))*m_soft).sum()/(m_soft.sum()+eps)*soft_weight
    return loss*weight

def chroma_gradient_preserve_loss(pred, tgt, lo=0.10, hi=0.90, min_grad=0.003, weight=1.0):
    eps = 1e-6
    lt  = _get_luma(tgt)
    t   = ((lt-lo)/(hi-lo+eps)).clamp(0.0,1.0)
    m_mid = t*t*(3.0-2.0*t)
    def uv(x):
        r,g,b = x[:,0:1],x[:,1:2],x[:,2:3]
        return -0.14713*r-0.28886*g+0.436*b, 0.615*r-0.51499*g-0.10001*b
    up,vp = uv(pred);  ut,vt = uv(tgt)
    def gm(c):
        return (c[:,:,:,1:]-c[:,:,:,:-1]).abs(),(c[:,:,1:,:]-c[:,:,:-1,:]).abs()
    up_h,up_v = gm(up);  ut_h,ut_v = gm(ut);  vp_h,vp_v = gm(vp);  vt_h,vt_v = gm(vt)
    mh = (m_mid[:,:,:,1:]*((ut_h+vt_h)>min_grad).float()).detach()
    mv = (m_mid[:,:,1:,:]*((ut_v+vt_v)>min_grad).float()).detach()
    loss_h = ((F.relu(ut_h-up_h)+F.relu(vt_h-vp_h))*mh).sum()/(mh.sum()+eps)
    loss_v = ((F.relu(ut_v-up_v)+F.relu(vt_v-vp_v))*mv).sum()/(mv.sum()+eps)
    return (loss_h+loss_v)*weight

def lightroom_rail_loss(pred, weight=15.0):
    return (F.relu(0.004-pred).pow(2)+F.relu(pred-0.996).pow(2)).mean()*weight

def hsl_regularization_loss(hsl_params, weight=0.02):
    l_sat = (hsl_params**2).mean()
    s_sat = ((hsl_params[:,1:]-hsl_params[:,:-1])**2).mean() + \
            ((hsl_params[:,0]-hsl_params[:,-1])**2).mean()
    return (l_sat + s_sat) * weight


def tv_loss_spatial(x):
    return ((x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean()+
            (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean())

def tv_loss_5d(x):
    return (torch.abs(x[:,:,1:]-x[:,:,:-1]).mean()+
            torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]).mean()+
            torch.abs(x[:,:,:,:,1:]-x[:,:,:,:,:-1]).mean())

def tv_loss_3d_weighted(x3d, w2d):
    B,C,D,Hg,Wg = x3d.shape
    w = F.interpolate(w2d,size=(Hg,Wg),mode="bilinear",align_corners=False).unsqueeze(2)
    return ((torch.abs(x3d[:,:,1:]-x3d[:,:,:-1])*w).mean()+
            (torch.abs(x3d[:,:,:,1:]-x3d[:,:,:,:-1])*w[:,:,:,1:]).mean()+
            (torch.abs(x3d[:,:,:,:,1:]-x3d[:,:,:,:,:-1])*w[:,:,:,:,1:]).mean())


class HueSatCurves(nn.Module):
    """YUV-space hue/sat curves (Hue shift + Saturation multiplier)."""
    def __init__(self, num_hues=12):
        super().__init__();  self.H = num_hues

    def forward(self, u, v, params):
        B, _, H, W = u.shape;  eps = 1e-8
        chroma  = torch.sqrt(u**2+v**2+eps)
        hue     = torch.atan2(v+eps, u+eps)
        hue_01  = (hue/(2.0*3.14159265359)+0.5)%1.0
        hue_pos = hue_01*self.H
        
        sat_raw = params[:, :self.H]
        hue_raw = params[:, self.H:]
        
        sat_mult = 1.0 + torch.tanh(sat_raw) * 0.75
        hue_shift = torch.tanh(hue_raw) * 0.35
        
        sat_pad  = torch.cat([sat_mult, sat_mult[:,:1]], dim=1)
        hue_pad  = torch.cat([hue_shift, hue_shift[:,:1]], dim=1)
        
        hue_pos  = hue_pos.clamp(0, self.H-1e-4)
        idx_lo   = hue_pos.floor().long();  idx_hi = idx_lo+1
        w_hi     = hue_pos-idx_lo.float();  w_lo   = 1.0-w_hi
        
        sat_pad  = sat_pad.view(B,self.H+1,1,1).expand(-1,-1,H,W)
        hue_pad  = hue_pad.view(B,self.H+1,1,1).expand(-1,-1,H,W)
        
        s_lo = torch.gather(sat_pad,1,idx_lo);  s_hi = torch.gather(sat_pad,1,idx_hi)
        sat_final  = s_lo*w_lo + s_hi*w_hi
        
        h_lo = torch.gather(hue_pad,1,idx_lo);  h_hi = torch.gather(hue_pad,1,idx_hi)
        hue_final  = hue + (h_lo*w_lo + h_hi*w_hi)
        
        chroma_new = chroma * sat_final
        u_new = chroma_new * torch.cos(hue_final)
        v_new = chroma_new * torch.sin(hue_final)
        
        blend = ((chroma-0.015)/(0.045-0.015+1e-8)).clamp(0.0,1.0)
        blend = blend*blend*(3.0-2.0*blend)
        u_new = u*(1.0-blend) + u_new*blend
        v_new = v*(1.0-blend) + v_new*blend
        return u_new, v_new


class GlobalChromaGrid3D(nn.Module):
    """3-D YUV bilateral grid for global color correction (doc-3)."""
    def __init__(self, luma_bins=12, uv_size=17, duv_max=0.25, dy_max=0.08):
        super().__init__()
        self.L=int(luma_bins); self.S=int(uv_size)
        self.duv_max=float(duv_max); self.dy_max=float(dy_max)

    def forward(self, y, u, v, grid):
        y_norm = (y*2.0-1.0).clamp(-1,1)
        u_norm = (u*2.0).clamp(-1,1)
        v_norm = (v*2.0).clamp(-1,1)
        coords = torch.stack([v_norm,u_norm,y_norm],dim=-1).squeeze(1).unsqueeze(1)
        delta  = F.grid_sample(grid,coords,mode='bilinear',
                               padding_mode='border',align_corners=True).squeeze(2)
        du = torch.tanh(delta[:,0:1])*self.duv_max
        dv = torch.tanh(delta[:,1:2])*self.duv_max
        dy = torch.tanh(delta[:,2:3])*self.dy_max
        return u+du, v+dv, dy
class GuidedColorGrid(nn.Module):
    def __init__(self, in_channels, grid_h=8, grid_w=8, depth=16,
                 guide_channels=32, max_uv=0.35):
        super().__init__()
        self.depth  = depth
        self.max_uv = max_uv

        self.guide_net = nn.Sequential(
            nn.Conv2d(in_channels + 2, 64, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, guide_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(guide_channels, guide_channels, 1),
        )

        self.grid_net = nn.Sequential(
            nn.Conv2d(in_channels + 2, 128, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 2 * depth, 1),
        )
        nn.init.zeros_(self.grid_net[-1].weight)
        nn.init.zeros_(self.grid_net[-1].bias)

        self.slice_net = nn.Sequential(
            nn.Conv2d(guide_channels, depth, 1),
        )
        nn.init.zeros_(self.slice_net[-1].weight)
        nn.init.zeros_(self.slice_net[-1].bias)

    def forward(self, feats, x_full, u_full, v_full):
        """
        feats:  (B, C, Hf, Wf) backbone features
        x_full: (B, 3, H, W)   full res image (for edge-aware upsample)
        u_full: (B, 1, H, W)   current U channel — color space position
        v_full: (B, 1, H, W)   current V channel — color space position
        """
        B, _, H, W = x_full.shape
        _, _, Hf, Wf = feats.shape

        u_small = F.interpolate(u_full, size=(Hf, Wf), mode='bilinear', align_corners=False)
        v_small = F.interpolate(v_full, size=(Hf, Wf), mode='bilinear', align_corners=False)

        feats_uv = torch.cat([feats, u_small, v_small], dim=1)

        raw_grid = self.grid_net(feats_uv)
        grid = raw_grid.view(B, 2, self.depth, Hf, Wf)
        grid = torch.tanh(grid) * self.max_uv

        guide_res = max(H // 8, Hf)
        feats_up  = F.interpolate(feats, size=(guide_res, guide_res),
                                  mode='bilinear', align_corners=False)
        u_guide   = F.interpolate(u_full, size=(guide_res, guide_res),
                                  mode='bilinear', align_corners=False)
        v_guide   = F.interpolate(v_full, size=(guide_res, guide_res),
                                  mode='bilinear', align_corners=False)
        feats_uv_guide = torch.cat([feats_up, u_guide, v_guide], dim=1)
        guide_feats    = self.guide_net(feats_uv_guide)

        slice_weights = self.slice_net(guide_feats)
        slice_weights = F.softmax(slice_weights, dim=1)

        grid_up = F.interpolate(
            grid.view(B, 2*self.depth, Hf, Wf),
            size=(guide_res, guide_res),
            mode='bilinear', align_corners=False
        ).view(B, 2, self.depth, guide_res, guide_res)

        sw = slice_weights.unsqueeze(1)
        uv_small = (grid_up * sw).sum(dim=2)

        luma_full = _get_luma(x_full)
        uv_offset = self._joint_bilateral_upsample(uv_small, luma_full, H, W)
        return uv_offset

    def _joint_bilateral_upsample(self, uv_small, luma_full, H, W):
        uv_up = F.interpolate(uv_small, size=(H, W),
                               mode='bilinear', align_corners=False)
        luma_small = F.interpolate(luma_full, size=uv_small.shape[-2:],
                                   mode='bilinear', align_corners=False)
        luma_small_up = F.interpolate(luma_small, size=(H, W),
                                       mode='bilinear', align_corners=False)
        luma_diff  = (luma_full - luma_small_up).abs()
        confidence = torch.exp(-luma_diff * 20.0)
        return uv_up * confidence

class Monotone1DCurve(nn.Module):
    def __init__(self, num_knots=16):
        super().__init__();  self.K=int(num_knots)

    def make_curve(self, raw_params):
        B = raw_params.shape[0]
        black_raw  = raw_params[:,0:1]
        slopes_raw = raw_params[:,1:]
        black_point = torch.sigmoid(black_raw)*0.025
        slopes = F.softplus(slopes_raw)+0.02
        c = torch.cumsum(slopes,dim=1)
        remaining = 1.0-black_point
        c = c/(c[:,-1:].clamp_min(1e-6))*remaining
        zeros = torch.zeros(B,1,device=raw_params.device,dtype=raw_params.dtype)
        curve = black_point+torch.cat([zeros,c],dim=1)
        return curve

    def apply(self, x01, curve):
        B,_,H,W = x01.shape
        x   = x01.clamp(0,1).view(B,1,-1)
        t   = x*(self.K-1)
        lo  = t.floor().long().clamp(0,self.K-2);  hi=lo+1
        w_hi = t-lo.float();  w_lo=1.0-w_hi
        curve = curve.view(B,1,self.K)
        v_lo  = torch.gather(curve,2,lo);  v_hi = torch.gather(curve,2,hi)
        return (w_lo*v_lo+w_hi*v_hi).view(B,1,H,W)

def shadow_slider_luma(
    y,
    shadows,
    blur_kernel=11,
    blur_sigma=2.2,
    shadow_lo=0.06,
    shadow_hi=0.42,
    black_floor=0.035,
    max_lift=0.16,
):
    """
    Lightroom-like shadow lift on luma only.

    y:       (B,1,H,W) luma in [0,1]
    shadows: (B,1,1,1) scalar control, expected roughly in [0,1]

    Behavior:
    - lifts blurred/base luma, not per-pixel raw luma
    - bounded to shadow/lower-mid range
    - protects deep blacks
    - re-adds local detail
    """
    s = shadows.clamp(0.0, 1.0)

    y_base = kornia.filters.gaussian_blur2d(y, (blur_kernel, blur_kernel), (blur_sigma, blur_sigma))
    detail = y - y_base

    black_protect = _smoothstep((y_base - black_floor) / (0.12 - black_floor + 1e-6))

    shadow_gate = 1.0 - _smoothstep((y_base - shadow_lo) / (shadow_hi - shadow_lo + 1e-6))
    shadow_gate = shadow_gate.clamp(0.0, 1.0)

    darkness = 1.0 - (y_base / (shadow_hi + 1e-6)).clamp(0.0, 1.0)
    darkness = darkness * darkness

    lift = s * max_lift * shadow_gate * darkness * black_protect
    y_base_lifted = y_base + lift

    detail_keep = 1.0 - (shadow_gate * s * 0.12)
    y_new = y_base_lifted + detail * detail_keep

    floor_anchor = 1.0 - _smoothstep((y - black_floor) / (0.08 - black_floor + 1e-6))
    y_new = y_new * (1.0 - floor_anchor * 0.45) + y * (floor_anchor * 0.45)

    return y_new.clamp(0.0, 1.0)
def shadow_slider_regularization_loss(shadows, weight=1.0):
    return (shadows.pow(2).mean()) * weight
def soft_highlight_shoulder_rgb(rgb, threshold=0.85, limit=0.996):
    """Per-channel shoulder — preserves color ratios through compression."""
    mask_lin = (rgb <= threshold).to(rgb.dtype)
    scale = limit - threshold
    overshoot = F.relu(rgb - threshold)
    shoulder = limit - scale * torch.exp(-overshoot / (scale + 1e-6))
    return rgb * mask_lin + shoulder * (1.0 - mask_lin)
def wb_global_cast_loss(
    x_wb,
    tgt,
    x_in=None,
    blur_sigma=10.0,
    luma_bins=(
        (0.06, 0.22, 0.70),
        (0.22, 0.60, 1.00),
        (0.60, 0.92, 0.75),
    ),
    chroma_min=0.010,
    overshoot_weight=0.35,
    magnitude_weight=0.25,
    input_anchor_weight=0.20,
    weight=1.0,
):
    """
    Low-frequency global illuminant / cast matching on the WB output.

    Purpose:
    - supervise temp/tint even when there are no neutral objects
    - match overall cool/warm and green/magenta scene cast
    - operate on blurred chroma so this targets illuminant feel, not local object color

    x_wb: image after WB temp/tint step
    tgt:  training target
    x_in: optional original input, used for a mild "move in correct direction" anchor
    """
    eps = 1e-6

    def ss(x, e0, e1):
        t = ((x - e0) / (e1 - e0 + eps)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def blur_rgb(x, sigma):
        k = int(max(3, round(sigma * 6))) | 1
        return kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma))

    def get_yuv(x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v =  0.615   * r - 0.51499 * g - 0.10001 * b
        c = torch.sqrt(u * u + v * v + eps)
        return y, u, v, c

    wb_b  = blur_rgb(x_wb, blur_sigma)
    tgt_b = blur_rgb(tgt,  blur_sigma)

    y_wb, u_wb, v_wb, c_wb = get_yuv(wb_b)
    y_t,  u_t,  v_t,  c_t  = get_yuv(tgt_b)

    if x_in is not None:
        in_b = blur_rgb(x_in, blur_sigma)
        _, u_in, v_in, c_in = get_yuv(in_b)
    else:
        u_in = v_in = c_in = None

    total = x_wb.new_tensor(0.0)
    total_w = x_wb.new_tensor(0.0)

    for lo, hi, band_w in luma_bins:
        band = ss(y_t, lo - 0.04, lo + 0.04) * (1.0 - ss(y_t, hi - 0.04, hi + 0.04))

        chroma_gate = 0.55 + 0.45 * ss(c_t, chroma_min, chroma_min + 0.05)

        m = (band * chroma_gate).detach()
        denom = m.sum(dim=(2, 3), keepdim=True)

        valid = (denom > 16.0).float()
        if valid.sum() < 1.0:
            continue

        mean_u_wb = (u_wb * m).sum(dim=(2, 3), keepdim=True) / (denom + eps)
        mean_v_wb = (v_wb * m).sum(dim=(2, 3), keepdim=True) / (denom + eps)

        mean_u_t  = (u_t * m).sum(dim=(2, 3), keepdim=True) / (denom + eps)
        mean_v_t  = (v_t * m).sum(dim=(2, 3), keepdim=True) / (denom + eps)

        du = mean_u_wb - mean_u_t
        dv = mean_v_wb - mean_v_t

        l1 = du.abs() + dv.abs()

        if x_in is not None:
            mean_u_in = (u_in * m).sum(dim=(2, 3), keepdim=True) / (denom + eps)
            mean_v_in = (v_in * m).sum(dim=(2, 3), keepdim=True) / (denom + eps)

            cross_u = ((mean_u_in - mean_u_t).sign() * (mean_u_wb - mean_u_t).sign() < 0).float()
            cross_v = ((mean_v_in - mean_v_t).sign() * (mean_v_wb - mean_v_t).sign() < 0).float()

            overshoot = du.abs() * cross_u + dv.abs() * cross_v

            base_err = (mean_u_in - mean_u_t).abs() + (mean_v_in - mean_v_t).abs()
            pred_err = du.abs() + dv.abs()
            anchor = F.relu(pred_err - base_err)
        else:
            overshoot = x_wb.new_tensor(0.0)
            anchor = x_wb.new_tensor(0.0)

        mag_wb = torch.sqrt(mean_u_wb * mean_u_wb + mean_v_wb * mean_v_wb + eps)
        mag_t  = torch.sqrt(mean_u_t  * mean_u_t  + mean_v_t  * mean_v_t  + eps)
        mag_loss = (mag_wb - mag_t).abs()

        band_loss = (
            l1 +
            overshoot * overshoot_weight +
            mag_loss * magnitude_weight +
            anchor * input_anchor_weight
        )

        total = total + (band_loss * valid).mean() * band_w
        total_w = total_w + band_w

    if total_w.item() <= 0.0:
        return x_wb.new_tensor(0.0)

    return (total / total_w) * weight
def wb_tonal_band_loss(x_wb, tgt, weight=1.0):
    """
    HQ Tonal Band WB: Ensures white balance holds across shadows, mids, and highlights.
    Minimizes the chromaticity (L1) of x_wb in target-neutral regions per band.
    """
    eps = 1e-6

    def get_yuv(x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v =  0.615   * r - 0.51499 * g - 0.10001 * b
        return y, u, v

    def ss(x, e0, e1):
        t = ((x - e0) / (e1 - e0 + eps)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    y_tgt, u_tgt, v_tgt = get_yuv(tgt)
    _, u_wb, v_wb = get_yuv(x_wb)

    chroma_tgt = torch.sqrt(u_tgt**2 + v_tgt**2 + eps)
    chroma_wb  = torch.sqrt(u_wb**2 + v_wb**2 + eps)

    neutral_mask = (1.0 - ss(chroma_tgt, 0.008, 0.040)).detach()

    bands =[
        (0.04, 0.12, 0.25, 0.35),
        (0.20, 0.30, 0.65, 0.75),
        (0.60, 0.72, 0.94, 0.98),
    ]

    total_loss = x_wb.new_tensor(0.0)

    for lo_start, lo_end, hi_start, hi_end in bands:
        band_mask = ss(y_tgt, lo_start, lo_end) * (1.0 - ss(y_tgt, hi_start, hi_end))
        m = (neutral_mask * band_mask).detach()

        pixel_count = m.sum(dim=(2, 3), keepdim=True)
        valid = (pixel_count > 32.0).float()

        band_chroma = (chroma_wb * m).sum(dim=(2, 3), keepdim=True) / (pixel_count + eps)
        
        total_loss = total_loss + (band_chroma * valid).mean()

    return total_loss * weight
def wb_alignment_loss(x_wb, tgt, weight=1.0):
    """
    HQ WB Alignment: Target-Guided Gray World.
    2-Pass Safe: Uses a small margin so it doesn't over-sterilize the image
    or cause oscillations on the second pass.
    """
    eps = 1e-6
    def get_yuv(x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v =  0.615   * r - 0.51499 * g - 0.10001 * b
        return y, u, v

    def ss(x, e0, e1):
        t = ((x - e0) / (e1 - e0 + eps)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    y_tgt, u_tgt, v_tgt = get_yuv(tgt)
    _, u_wb, v_wb = get_yuv(x_wb)

    chroma_tgt = torch.sqrt(u_tgt**2 + v_tgt**2 + eps)
    chroma_wb  = torch.sqrt(u_wb**2 + v_wb**2 + eps)
    
    neutral_mask = 1.0 - ss(chroma_tgt, 0.005, 0.035)
    luma_gate = ss(y_tgt, 0.05, 0.15) * (1.0 - ss(y_tgt, 0.90, 0.98))
    
    mask = (neutral_mask * luma_gate).detach()
    pixel_count = mask.sum(dim=(2, 3), keepdim=True)
    has_neutrals = (pixel_count > 64.0).float()

    chroma_penalty = F.relu(chroma_wb - 0.005)
    neutral_loss = (chroma_penalty * mask).sum(dim=(2, 3), keepdim=True) / (pixel_count + eps)

    uv_tgt_vec = torch.cat([u_tgt, v_tgt], dim=1)
    uv_wb_vec  = torch.cat([u_wb, v_wb], dim=1)
    cos_sim = F.cosine_similarity(uv_wb_vec, uv_tgt_vec, dim=1, eps=eps).unsqueeze(1)
    
    color_mask  = (ss(chroma_tgt, 0.02, 0.10) * luma_gate).detach()
    color_count = color_mask.sum(dim=(2, 3), keepdim=True)
    
    fallback_loss = (F.relu(0.99 - cos_sim) * color_mask).sum(dim=(2, 3), keepdim=True) / (color_count + eps)

    final_loss = has_neutrals * neutral_loss + (1.0 - has_neutrals) * fallback_loss

    return final_loss.mean() * weight
def colored_highlight_anti_desat_loss(
    pred,
    tgt,
    x_in,
    core_thresh=0.985,
    ring_radius=6,
    bright_lo=0.70,
    bright_hi=0.98,
    target_chroma_lo=0.020,
    input_chroma_lo=0.030,
    input_fallback_weight=0.35,
    under_margin=0.003,
    under_weight=1.0,
    over_weight=0.05,
    hue_weight=0.20,
    weight=1.0,
):
    """
    Protects colored highlight-adjacent regions from being washed out.

    Design:
    - highlight/risk region comes from INPUT hot cores / near-clipped areas
    - color evidence comes mainly from TARGET, with a small INPUT fallback
    - only strongly penalizes UNDER-chroma (desaturation)
    - only lightly penalizes OVER-chroma
    - lightly preserves hue direction
    - does NOT define the mask from prediction, so the model cannot escape
      by already desaturating the region

    This is intended for cases like:
    white core + orange/yellow/cyan glow ring around bright light sources.
    """
    eps = 1e-6

    def get_yuv(x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v =  0.615   * r - 0.51499 * g - 0.10001 * b
        c = torch.sqrt(u * u + v * v + eps)
        return y, u, v, c

    def ss(x, e0, e1):
        t = ((x - e0) / (e1 - e0 + eps)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    yi, ui, vi, ci = get_yuv(x_in)
    yt, ut, vt, ct = get_yuv(tgt)
    yp, up, vp, cp = get_yuv(pred)

    max_rgb_in = x_in.amax(dim=1, keepdim=True)
    core = (max_rgb_in >= core_thresh).float()

    if core.sum() < 1.0:
        return pred.new_tensor(0.0)

    k = ring_radius * 2 + 1
    dilated = F.max_pool2d(core, kernel_size=k, stride=1, padding=ring_radius)
    ring = (dilated - core).clamp(0.0, 1.0)

    near_core = F.avg_pool2d(dilated, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)

    y_ref = torch.maximum(yi, yt)
    bright_gate = ss(y_ref, bright_lo, bright_hi)

    color_t = ss(ct, target_chroma_lo, target_chroma_lo + 0.05)
    color_i = ss(ci, input_chroma_lo,  input_chroma_lo  + 0.06)

    color_gate = torch.maximum(color_t, color_i * input_fallback_weight)

    chroma_weight = (1.0 + 1.5 * ss(ct, 0.04, 0.14)).detach()

    m = (near_core * ring * bright_gate * color_gate).detach()

    denom = m.sum()
    if denom < 1.0:
        return pred.new_tensor(0.0)

    c_ref = torch.maximum(ct, ci * input_fallback_weight)

    under = F.relu((c_ref - under_margin) - cp)

    over = F.relu(cp - (c_ref + 0.02))

    chroma_loss = (
        ((under_weight * under) + (over_weight * over)) * chroma_weight * m
    ).sum() / (denom + eps)

    uv_pred = torch.cat([up, vp], dim=1)
    uv_tgt  = torch.cat([ut, vt], dim=1)

    cos_sim = F.cosine_similarity(uv_pred, uv_tgt, dim=1, eps=1e-8).unsqueeze(1)

    hue_gate = (m * ss(ct, 0.03, 0.10)).detach()
    hue_denom = hue_gate.sum()

    if hue_denom > 1.0:
        hue_loss = ((1.0 - cos_sim) * hue_gate).sum() / (hue_denom + eps)
    else:
        hue_loss = pred.new_tensor(0.0)

    return (chroma_loss + hue_weight * hue_loss) * weight
class BilateralGridEditor(nn.Module):

    def __init__(self, backbone_name='mobilenetv4_conv_small.e2400_r224_in1k',
                 grid_d=24, pretrained=True):
        super().__init__()
        self.grid_d = grid_d

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            features_only=True, out_indices=[2, 3, -1]
        )
        enc_ch_list = self.backbone.feature_info.channels()
        enc_channels = enc_ch_list[-1]
        total_pooled = sum(c*9 for c in enc_ch_list)

        self.K        = 16
        self.stats_dim = 61

        OUT_DIM = 1 + 1 + self.K + 2 + 1

        self.global_head = nn.Sequential(
            nn.Linear(total_pooled + self.stats_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, OUT_DIM),
        )
        with torch.no_grad():
            self.global_head[-1].weight.zero_()
            self.global_head[-1].bias.zero_()
            self.global_head[-1].bias.data[1] = -6.0
            self.global_head[-1].bias.data[2 + self.K + 2] = -4.0

        self.luma_curve = Monotone1DCurve(num_knots=self.K)

        self.chroma_grid_bins = 10
        self.chroma_grid_uv_size = 13
        self.chroma_grid = GlobalChromaGrid3D(
            luma_bins=self.chroma_grid_bins,
            uv_size=self.chroma_grid_uv_size,
            duv_max=0.10,
            dy_max=0.06,
        )
        chroma_grid_out = 3 * self.chroma_grid_bins * self.chroma_grid_uv_size * self.chroma_grid_uv_size
        self.chroma_grid_head = nn.Sequential(
            nn.Conv2d(enc_channels, 256, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, chroma_grid_out, 1),
        )
        nn.init.zeros_(self.chroma_grid_head[-1].weight)
        nn.init.zeros_(self.chroma_grid_head[-1].bias)

        self.a_max = 0.80;  self.b_max = 0.40

        self.grid_channels = 2 * grid_d
        self.grid_head = nn.Sequential(
            nn.Conv2d(enc_channels, 256, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, self.grid_channels, 1),
        )
        nn.init.zeros_(self.grid_head[-1].weight)
        nn.init.zeros_(self.grid_head[-1].bias)


    def forward(self, x_full, x384):
        bs   = x384.shape[0]
        H, W = x_full.shape[-2:]

        feat_list = self.backbone(x384)
        feats     = feat_list[-1]

        pooled_parts = [F.adaptive_avg_pool2d(f,3).flatten(1) for f in feat_list]
        pooled       = torch.cat(pooled_parts, dim=1)

        stats = image_stats_10(x_full).to(pooled.dtype)
        p     = self.global_head(torch.cat([pooled, stats], dim=1))

        ev_raw    = p[:,0:1]
        hi_raw    = p[:,1:2]
        curve_raw = p[:,2:2+self.K]

        wb_start  = 2 + self.K
        temp = torch.tanh(p[:, wb_start:wb_start+1]).view(bs, 1, 1, 1)
        tint = torch.tanh(p[:, wb_start+1:wb_start+2]).view(bs, 1, 1, 1)

        shadow_raw = p[:, wb_start+2:wb_start+3]
        shadows = torch.sigmoid(shadow_raw).view(bs, 1, 1, 1) * 0.50

        x_wb = apply_wb_temp_tint_bradford(
            x_full, temp=temp, tint=tint,
            base_cct=6500.0, temp_mired_range=120.0,
            tint_uv_range=0.06, preserve_luma=True,
        )

        yuv  = rgb_to_yuv_bt601(x_wb)
        y    = yuv[:,0:1]; u = yuv[:,1:2]; v = yuv[:,2:3]

        ev      = torch.tanh(ev_raw) * 0.90
        if ev.dim() == 2:
            ev = ev[:, :1].view(bs, 1, 1, 1)

        ev_fade = 1.0 - ((y - 0.55) / (0.90 - 0.55 + 1e-6)).clamp(0.0, 1.0)
        ev_fade = ev_fade * ev_fade * (3.0 - 2.0 * ev_fade)
        y       = soft_rolloff(y * torch.exp(ev * ev_fade), knee=0.94, limit=1.12)

        y = shadow_slider_luma(
            y,
            shadows=shadows,
            blur_kernel=11,
            blur_sigma=2.2,
            shadow_lo=0.06,
            shadow_hi=0.42,
            black_floor=0.035,
            max_lift=0.16,
        )

        hi      = torch.sigmoid(hi_raw).view(bs,1,1,1)
        rgb_int = yuv_to_rgb_bt601(torch.cat([y,u,v],dim=1))
        y       = dynamic_highlight_recovery(y, rgb_int, hi, threshold=0.85)

        curve           = self.luma_curve.make_curve(curve_raw)
        luma_for_detail = y.detach().clone()
        y               = self.luma_curve.apply(y, curve)

        yuv2 = rgb_to_yuv_bt601(yuv_to_rgb_bt601(torch.cat([y,u,v],dim=1)))
        y_in = yuv2[:,0:1]

        raw_grid    = self.grid_head(feats)
        raw_grid    = kornia.filters.gaussian_blur2d(raw_grid,(3,3),(0.5,0.5))
        _,_,h_g,w_g = raw_grid.shape
        grid_5d     = raw_grid.view(bs,2,self.grid_d,h_g,w_g)
        grid_t      = torch.tanh(grid_5d)
        a_grid      = grid_t[:,0:1]*self.a_max
        b_grid      = grid_t[:,1:2]*self.b_max

        y_guide  = kornia.filters.gaussian_blur2d(y_in,(3,3),(0.6,0.6))
        guidance = y_guide.clamp(0,1)*2.0-1.0
        y_coords = torch.linspace(-1,1,H,device=x_full.device).view(1,1,H,1).expand(bs,-1,-1,W)
        x_coords = torch.linspace(-1,1,W,device=x_full.device).view(1,1,1,W).expand(bs,-1,H,-1)
        coords   = torch.stack((x_coords,y_coords,guidance),dim=-1)

        coeffs  = F.grid_sample(torch.cat([a_grid,b_grid],dim=1),coords,
                                mode='bilinear',padding_mode='border',
                                align_corners=True).squeeze(2)
        a_map = coeffs[:,0:1]; b_map = coeffs[:,1:2]

        t_shadow    = ((y_in-0.03)/(0.12-0.03+1e-6)).clamp(0.0,1.0)
        shadow_fade = t_shadow*t_shadow*(3.0-2.0*t_shadow)
        a_map = a_map*shadow_fade
        b_map = b_map*shadow_fade

        y_out = y_in * (1.0 + a_map) + b_map

        rgb_pre_shoulder = yuv_to_rgb_bt601(
            torch.cat([y_out, yuv2[:, 1:2], yuv2[:, 2:3]], dim=1)
        )

        rgb_shouldered = soft_highlight_shoulder_rgb(rgb_pre_shoulder)

        yuv_pre_chroma = rgb_to_yuv_bt601(rgb_shouldered)

        yuv_pre_chroma = restore_highlight_detail_pointwise(
            yuv_pre_chroma, luma_for_detail
        )

        y_final = yuv_pre_chroma[:,0:1]
        u_final = yuv_pre_chroma[:,1:2]
        v_final = yuv_pre_chroma[:,2:3]

        L  = self.chroma_grid_bins
        S  = self.chroma_grid_uv_size
        raw_cg = self.chroma_grid_head(feats)
        raw_cg = F.adaptive_avg_pool2d(raw_cg, 1).view(bs, 3, L, S, S)
        u_final, v_final, dy = self.chroma_grid(y_final, u_final, v_final, raw_cg)
        y_final = (y_final + dy).clamp(0.004, 0.996)
        u_final = u_final.clamp(-0.60, 0.60)
        v_final = v_final.clamp(-0.60, 0.60)

        out = yuv_to_rgb_bt601(torch.cat([y_final, u_final, v_final], dim=1)).clamp(0.004, 0.996)

        aux = {
            "out_raw":      out,
            "a_grid":       a_grid,
            "b_grid":       b_grid,
            "chroma_grid":  raw_cg,
            "curve":        curve,
            "shadows":      shadows,
            "ev":           ev,
            "hi":           hi,
            "temp":         temp,
            "tint":         tint,
            "x_wb":         x_wb,
            "x_in":         x_full,
        }
        return out, aux

    
    @torch.no_grad()
    def apply_params_chunked(self, xfull, params, chunk_h=256, chunk_w=512):
        return self.apply_params(
            xfull,
            params,
            chunk_h=chunk_h,
            chunk_w=chunk_w,
        )
    @torch.no_grad()
    def apply_params(
        self,
        xfull,
        params,
        coord_offset=None,
        full_size=None,
        chunk_h=None,
        chunk_w=None,
    ):
        """
        Unified apply function.

        Modes:
        1) Full image:
            apply_params(xfull, params)

        2) Tile with global coordinate mapping:
            apply_params(x_tile, params, coord_offset=(y0, x0), full_size=(H_full, W_full))

        3) Automatic chunked full-image processing:
            apply_params(xfull, params, chunk_h=256, chunk_w=512)

        Important:
        - Uses the exact same operation order as training forward.
        - No duplicated processing logic.
        """
        B, C, H, W = xfull.shape

        if chunk_h is not None and chunk_w is not None and (H > chunk_h or W > chunk_w):
            out = torch.empty_like(xfull)

            for y0 in range(0, H, chunk_h):
                y1 = min(y0 + chunk_h, H)
                for x0 in range(0, W, chunk_w):
                    x1 = min(x0 + chunk_w, W)

                    tile = xfull[:, :, y0:y1, x0:x1]
                    out[:, :, y0:y1, x0:x1] = self.apply_params(
                        tile,
                        params,
                        coord_offset=(y0, x0),
                        full_size=(H, W),
                        chunk_h=None,
                        chunk_w=None,
                    )
            return out

        bs = xfull.shape[0]

        ev      = params["ev"]
        hi      = params["hi"]
        curve   = params["curve"]
        a_grid  = params["a_grid"]
        b_grid  = params["b_grid"]
        temp    = params["temp"]
        tint    = params["tint"]
        shadows = params["shadows"]
        chroma_grid_raw = params["chroma_grid"]

        x_wb = apply_wb_temp_tint_bradford(
            xfull,
            temp=temp,
            tint=tint,
            base_cct=6500.0,
            temp_mired_range=120.0,
            tint_uv_range=0.06,
            preserve_luma=True,
        )

        yuv = rgb_to_yuv_bt601(x_wb)
        y, u, v = yuv[:, 0:1], yuv[:, 1:2], yuv[:, 2:3]

        ev_fade = 1.0 - ((y - 0.55) / (0.90 - 0.55 + 1e-6)).clamp(0.0, 1.0)
        ev_fade = ev_fade * ev_fade * (3.0 - 2.0 * ev_fade)
        y = soft_rolloff(y * torch.exp(ev * ev_fade), knee=0.94, limit=1.12)

        y = shadow_slider_luma(
            y,
            shadows=shadows,
            blur_kernel=11,
            blur_sigma=2.2,
            shadow_lo=0.06,
            shadow_hi=0.42,
            black_floor=0.035,
            max_lift=0.16,
        )

        rgb_int = yuv_to_rgb_bt601(torch.cat([y, u, v], dim=1))
        y = dynamic_highlight_recovery(y, rgb_int, hi, threshold=0.85)

        luma_for_detail = y.detach().clone()
        y = self.luma_curve.apply(y, curve)

        yuv2 = rgb_to_yuv_bt601(yuv_to_rgb_bt601(torch.cat([y, u, v], dim=1)))
        y_in = yuv2[:, 0:1]

        y_guide  = kornia.filters.gaussian_blur2d(y_in, (3, 3), (0.6, 0.6))
        guidance = y_guide.clamp(0, 1) * 2.0 - 1.0

        if coord_offset is not None and full_size is not None:
            y0_off, x0_off = coord_offset
            H_full, W_full = full_size

            y_pixels = torch.arange(
                y0_off, y0_off + H, device=xfull.device, dtype=xfull.dtype
            )
            x_pixels = torch.arange(
                x0_off, x0_off + W, device=xfull.device, dtype=xfull.dtype
            )

            y_norm = (y_pixels / max(H_full - 1, 1)) * 2.0 - 1.0
            x_norm = (x_pixels / max(W_full - 1, 1)) * 2.0 - 1.0

            y_coords = y_norm.view(1, 1, H, 1).expand(bs, -1, -1, W)
            x_coords = x_norm.view(1, 1, 1, W).expand(bs, -1, H, -1)
        else:
            y_coords = torch.linspace(-1, 1, H, device=xfull.device).view(1, 1, H, 1).expand(bs, -1, -1, W)
            x_coords = torch.linspace(-1, 1, W, device=xfull.device).view(1, 1, 1, W).expand(bs, -1, H, -1)

        coords = torch.stack((x_coords, y_coords, guidance), dim=-1)

        coeffs = F.grid_sample(
            torch.cat([a_grid, b_grid], dim=1),
            coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(2)

        a_map = coeffs[:, 0:1]
        b_map = coeffs[:, 1:2]

        t_shadow = ((y_in - 0.03) / (0.12 - 0.03 + 1e-6)).clamp(0.0, 1.0)
        shadow_fade = t_shadow * t_shadow * (3.0 - 2.0 * t_shadow)
        a_map = a_map * shadow_fade
        b_map = b_map * shadow_fade

        y_out = y_in * (1.0 + a_map) + b_map

        rgb_pre_shoulder = yuv_to_rgb_bt601(
            torch.cat([y_out, yuv2[:, 1:2], yuv2[:, 2:3]], dim=1)
        )

        rgb_shouldered = soft_highlight_shoulder_rgb(rgb_pre_shoulder)

        yuv_pre_chroma = rgb_to_yuv_bt601(rgb_shouldered)

        yuv_pre_chroma = restore_highlight_detail_pointwise(
            yuv_pre_chroma, luma_for_detail
        )

        y_final = yuv_pre_chroma[:, 0:1]
        u_final = yuv_pre_chroma[:, 1:2]
        v_final = yuv_pre_chroma[:, 2:3]

        u_final, v_final, dy = self.chroma_grid(y_final, u_final, v_final, chroma_grid_raw)

        y_final = (y_final + dy).clamp(0.004, 0.996)
        u_final = u_final.clamp(-0.60, 0.60)
        v_final = v_final.clamp(-0.60, 0.60)

        out = yuv_to_rgb_bt601(torch.cat([y_final, u_final, v_final], dim=1)).clamp(0.004, 0.996)
        return out
def wb_temp_tint_regularization_loss(temp, tint, weight=1.0):
    return (temp.pow(2).mean() * 0.25 + tint.pow(2).mean() * 1.0) * weight

def shadow_detail_preserve_loss(pred, x_in, dark_lo=0.0, dark_hi=0.20, margin=0.4, weight=1.0):
    eps = 1e-6
    lp = _get_luma(pred)
    lx = _get_luma(x_in)

    t = ((lp - dark_lo) / (dark_hi - dark_lo + eps)).clamp(0.0, 1.0)
    m_sh = (1.0 - t * t * (3.0 - 2.0 * t)).detach()
    
    total = pred.new_tensor(0.0)
    denom = pred.new_tensor(0.0)
    
    scales = [
        (5,  0.020, 0.012),
        (15, 0.015, 0.010),
    ]
    
    for k, min_input_std, min_std_loss in scales:
        s = k / 2.5
        k_size = k if k % 2 == 1 else k + 1
        
        mean_x = kornia.filters.gaussian_blur2d(lx, (k_size, k_size), (s, s))
        mean_x2 = kornia.filters.gaussian_blur2d(lx * lx, (k_size, k_size), (s, s))
        std_x = torch.sqrt((mean_x2 - mean_x * mean_x).clamp_min(0.0) + eps)
        
        mean_p = kornia.filters.gaussian_blur2d(lp, (k_size, k_size), (s, s))
        mean_p2 = kornia.filters.gaussian_blur2d(lp * lp, (k_size, k_size), (s, s))
        std_p = torch.sqrt((mean_p2 - mean_p * mean_p).clamp_min(0.0) + eps)
        
        has_detail = _smoothstep((std_x - min_input_std * 0.5) / (min_input_std * 0.5 + eps))
        
        std_loss = std_x - std_p
        is_losing = _smoothstep((std_loss - min_std_loss * 0.5) / (min_std_loss * 0.5 + eps))
        
        ratio = std_p / (std_x + eps)
        is_crushed = 1.0 - _smoothstep((ratio - 0.55) / (0.20 + eps))
        
        gate = (has_detail * is_losing * is_crushed).detach()
        
        allowed_std = std_x * (1.0 - margin)
        crush_penalty = F.relu(allowed_std - std_p)
        
        m = m_sh * gate
        total = total + (crush_penalty * m).sum()
        denom = denom + m.sum()
    
    if denom < 1.0:
        return pred.new_tensor(0.0)
    
    return (total / (denom + eps)) * weight
def local_tonal_contrast_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    x_orig: torch.Tensor,
    luma_lo: float = 0.10,
    luma_hi: float = 0.92,
    amp_floor: float = 0.0025,
    under_weight: float = 1.00,
    over_weight: float = 0.30,
    sign_weight: float = 0.35,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    Improved local tonal contrast loss.

    Main fix:
    compares LOCAL CONTRAST MAGNITUDE, not only signed detail.

    Why:
    the old version can miss flattening on the dark side of a shading transition.
    Example:
      target detail = -0.030
      pred   detail = -0.010

    That is clearly flattened, but signed subtraction treats it too softly.

    This version:
    - compares abs(detail) amplitudes
    - still lightly preserves sign / phase
    - uses smooth gating
    """
    eps = 1e-6

    lp = _get_luma(pred)
    lt = _get_luma(tgt)
    lo = _get_luma(x_orig)

    t_lo = ((lt - luma_lo) / (0.08 + eps)).clamp(0.0, 1.0)
    t_hi = ((lt - luma_hi) / (0.05 + eps)).clamp(0.0, 1.0)
    gate_luma = (
        t_lo * t_lo * (3.0 - 2.0 * t_lo) *
        (1.0 - t_hi * t_hi * (3.0 - 2.0 * t_hi))
    ).detach()

    total = pred.new_tensor(0.0)

    scales = [
        (4.0, 12.0, 0.6),
        (2.5,  7.0, 0.3),
        (1.5,  4.5, 0.1),
    ]

    for (s_inner, s_outer, sw) in scales:
        k_inner = int(max(3, round(s_inner * 6))) | 1
        k_outer = int(max(3, round(s_outer * 6))) | 1

        blur_t_in = kornia.filters.gaussian_blur2d(lt, (k_inner, k_inner), (s_inner, s_inner))
        blur_t_out = kornia.filters.gaussian_blur2d(lt, (k_outer, k_outer), (s_outer, s_outer))
        blur_p_in = kornia.filters.gaussian_blur2d(lp, (k_inner, k_inner), (s_inner, s_inner))
        blur_p_out = kornia.filters.gaussian_blur2d(lp, (k_outer, k_outer), (s_outer, s_outer))
        blur_o_in = kornia.filters.gaussian_blur2d(lo, (k_inner, k_inner), (s_inner, s_inner))
        blur_o_out = kornia.filters.gaussian_blur2d(lo, (k_outer, k_outer), (s_outer, s_outer))

        detail_t = blur_t_in - blur_t_out
        detail_p = blur_p_in - blur_p_out
        detail_o = blur_o_in - blur_o_out

        amp_t = detail_t.abs()
        amp_o = detail_o.abs()
        amp_p = detail_p.abs()

        use_target = (amp_t >= amp_o).float()
        ref_detail = use_target * detail_t + (1.0 - use_target) * detail_o
        ref_amp = ref_detail.abs()

        detail_gate = _smoothstep((ref_amp - amp_floor) / (amp_floor + eps))
        gate = (gate_luma * detail_gate).detach()

        denom = gate.sum()
        if denom < 1.0:
            continue

        amp_under = F.relu(ref_amp - amp_p)

        amp_over = F.relu(amp_p - ref_amp)

        sign_mismatch = ((ref_detail * detail_p) < 0).float() * torch.minimum(ref_amp, amp_p)

        loss_map = (
            under_weight * amp_under +
            over_weight * amp_over +
            sign_weight * sign_mismatch
        )

        total = total + (loss_map * gate).sum() / (denom + eps) * sw

    return total * weight



class ColorEnhancementLosses(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.weights     = dict(BASE_LOSS_WEIGHTS)
        self.charbonnier = Charbonnier(eps=1e-3)

    def _loss_global_reg(self, aux, need_gate):
        curve  = aux["curve"]
        slopes = curve[:,1:]-curve[:,:-1]
        loss_bounds = F.relu(0.005-slopes).mean()+F.relu(slopes-0.30).mean()
        d2          = slopes[:,1:]-slopes[:,:-1]
        loss_smooth = (d2**2).mean()*0.5
        return ((loss_bounds*0.3)+loss_smooth)*need_gate.detach().mean()

    def forward(self, pred, tgt, x_in, aux, gate_score,
                original_input=None):
        x_ref   = original_input if original_input is not None else x_in
        lp = _get_luma(pred);  lt = _get_luma(tgt);  lx = _get_luma(x_in)
        lx_orig = _get_luma(original_input) if original_input is not None else None

        losses = {}

        losses["rgb"]     = self.charbonnier(pred, tgt)
        losses["ms_ssim"] = multiscale_luma_charbonnier_loss(pred, tgt)
        
        t_lo = _smoothstep((lt-0.05)/(0.05+1e-6))
        t_hi = _smoothstep((lt-0.95)/(0.04+1e-6))
        m_luma = (t_lo*(1.0-t_hi)).detach()

        upper_mid_gate = _smoothstep((lt - 0.68) / (0.08 + 1e-6)) * (1.0 - _smoothstep((lt - 0.88) / (0.05 + 1e-6)))
        overshoot_scale = (1.0 + upper_mid_gate * 1.5).detach()

        losses["luma_match"] = toward_target_loss(
            lp, lt, lx,
            x_orig=lx_orig,
            mask=m_luma,
            overshoot_weight=0.35,
            regression_weight=1.15,
            completion_weight=0.18,
            overshoot_scale=overshoot_scale,
        )

        losses["input_hi_brake"] = input_highlight_brake_loss(pred, x_ref)
        losses["hi_grad_preserve"] = highlight_gradient_preserve_loss(
            pred, x_orig=x_ref, tgt=tgt, lo=0.80, weight=1.0)
        losses["colored_hi_anti_desat"] = colored_highlight_anti_desat_loss(
            pred, tgt, x_ref,
            core_thresh=0.985,
            ring_radius=6,
            bright_lo=0.70,
            bright_hi=0.98,
            target_chroma_lo=0.020,
            input_chroma_lo=0.030,
            input_fallback_weight=0.35,
            under_margin=0.003,
            under_weight=1.0,
            over_weight=0.05,
            hue_weight=0.20,
            weight=1.0,
        )
        losses["hue_band_chroma_v2"] = hue_band_chroma_match_loss_v2(
            pred, tgt,
            luma_lo=0.03,
            luma_hi=0.97,
            chroma_lo=0.018,
            chroma_hi=0.28,
            num_hues=8,
            under_weight=1.0,
            over_weight=0.75,
            hue_weight=0.16,
            downsample=2,
            weight=1.0,
        )
        losses["lower_mid_tone_push"] = lower_mid_tone_push_loss(
            pred, tgt, x_ref,
            lo=0.14,
            hi=0.55,
            under_weight=1.20,
            over_weight=0.40,
            detail_ref_mix=0.5,
        )
        losses["shadow_density"] = shadow_density_loss(pred, tgt)
        losses["shadow_reg"] = shadow_slider_regularization_loss(aux["shadows"])
        losses["shadow_detail"] = shadow_detail_preserve_loss(pred, x_ref)
        losses["yuv_match"] = full_range_uv_match_loss(
            pred, tgt, x_in, x_orig=original_input)
        losses["uv_energy"] = uv_energy_loss(pred, tgt)
        losses["wb_alignment"] = wb_alignment_loss(
            aux["x_wb"], tgt, weight=1.0)
        losses["wb_global_cast"] = wb_global_cast_loss(
            aux["x_wb"],
            tgt,
            x_in=x_ref,
            blur_sigma=10.0,
            luma_bins=(
                (0.06, 0.22, 0.70),
                (0.22, 0.60, 1.00),
                (0.60, 0.92, 0.75),
            ),
            chroma_min=0.010,
            overshoot_weight=0.35,
            magnitude_weight=0.25,
            input_anchor_weight=0.20,
            weight=1.0,
        )
        losses["wb_tonal_band"] = wb_tonal_band_loss(aux["x_wb"], tgt)
        losses["wb_reg"] = wb_temp_tint_regularization_loss(
            aux["temp"], aux["tint"])

        losses["input_neutral_preserve"] = input_neutral_preserve_loss(
            pred, x_ref, tgt)
        losses["target_neutral_preserve"] = target_neutral_preserve_loss(pred, tgt)
       
        losses["detail_preserve"] = full_range_detail_preserve_loss(
            pred, tgt, x_ref)
        losses["chroma_grad"] = chroma_gradient_preserve_loss(pred, tgt)
        losses["local_tonal_contrast"] = local_tonal_contrast_loss(
            pred, tgt, x_ref)
  
        out_raw = aux.get("out_raw", pred)
        losses["raw_overload"]    = (F.relu(out_raw-1.0).pow(2)+
                                     F.relu(0.0-out_raw).pow(2)).mean()
        losses["lightroom_rails"] = lightroom_rail_loss(pred, weight=1.0)

        if torch.is_tensor(gate_score):
            reg_gate = gate_score.detach().mean().clamp(0.0,1.0)
        else:
            reg_gate = pred.new_tensor(float(gate_score)).clamp(0.0,1.0)
        reg_gate = reg_gate**1.2

        m_hi_detail = _smoothstep((lt-0.8)/0.2)
        w_tv = (0.25+0.75*(1.0-m_hi_detail)).clamp(0.25,1.0)

        losses["tv_grid"] = (tv_loss_3d_weighted(aux["a_grid"],w_tv)+
                             tv_loss_3d_weighted(aux["b_grid"],w_tv))*reg_gate

        cg = aux["chroma_grid"]
        losses["tv_chroma_grid"] = tv_loss_5d(cg)

        losses["global_reg"] = self._loss_global_reg(aux, gate_score)

        total = pred.new_tensor(0.0)
        for k, v in losses.items():
            if k in self.weights:
                total = total + v * self.weights[k]
        return total, losses


class GlobalBucketDataset(Dataset):
    def __init__(self, root, aug=True):
        self.aug     = aug
        self.samples = []
        bucket_dirs  = sorted([p for p in root.iterdir() if p.is_dir()],
                               key=lambda p: p.name)
        if not bucket_dirs:
            raise RuntimeError(f"No bucket subfolders found in: {root}")
        for bdir in bucket_dirs:
            for f in sorted(bdir.glob("*.pt")):
                self.samples.append((f, bdir.name))
        if not self.samples:
            raise RuntimeError(f"No .pt files found under: {root}")
        self.bucket_names  = sorted({b for _,b in self.samples})
        self.bucket_to_id  = {b:i for i,b in enumerate(self.bucket_names)}
        self.bucket_indices = {b:[] for b in self.bucket_names}
        for idx,(_,bname) in enumerate(self.samples):
            self.bucket_indices[bname].append(idx)

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        pt_path, bucket_name = self.samples[idx]
        x, y = torch.load(pt_path, weights_only=True)
        x = x.float()/255.0;  y = y.float()/255.0
        is_identity = "__identity" in pt_path.stem
        is_cmaug    = pt_path.stem.startswith("cmaug_")
        if self.aug:
            if torch.rand(1).item() < 0.50:
                x = torch.flip(x, [-1])
                y = torch.flip(y, [-1])

            x, y = paired_crop_scale_jitter(x, y, p=0.25, min_scale=0.94)

            if not is_identity and not is_cmaug:
                with torch.no_grad():
                    x = tone_aug_x_only(x, x_target=y)
        return x, y, self.bucket_to_id[bucket_name]


class BucketBatchSampler(torch.utils.data.Sampler):
    def __init__(self, bucket_indices, batch_size, shuffle, seed=42):
        self.bucket_indices = {k:list(v) for k,v in bucket_indices.items()}
        self.bucket_names   = list(self.bucket_indices.keys())
        self.batch_size     = int(batch_size)
        self.shuffle        = bool(shuffle)
        self.rng            = random.Random(seed)
        self.bucket_num_batches = {b:len(idxs)//self.batch_size
                                   for b,idxs in self.bucket_indices.items()}
        self.total_batches  = sum(self.bucket_num_batches.values())
        if self.total_batches<=0:
            raise RuntimeError("No full batches available.")
        weights = [max(1,self.bucket_num_batches[b]) for b in self.bucket_names]
        s = sum(weights)
        self.bucket_weights = weights
        self.bucket_probs   = [w/s for w in weights]

    def __len__(self):  return self.total_batches

    def __iter__(self):
        work = {}
        for b in self.bucket_names:
            idxs = list(self.bucket_indices[b])
            if self.shuffle: self.rng.shuffle(idxs)
            work[b] = idxs
        ptr          = {b:0 for b in self.bucket_names}
        batches_left = dict(self.bucket_num_batches)
        for _ in range(self.total_batches):
            chosen = None
            for _ in range(10):
                b = self.rng.choices(self.bucket_names,
                                     weights=self.bucket_weights, k=1)[0]
                if batches_left[b]>0: chosen=b; break
            if chosen is None:
                for b in self.bucket_names:
                    if batches_left[b]>0: chosen=b; break
            start = ptr[chosen];  end = start+self.batch_size
            yield work[chosen][start:end]
            ptr[chosen] = end;  batches_left[chosen] -= 1

def bucket_collate(batch):
    xs,ys,bs = zip(*batch)
    return torch.stack(xs), torch.stack(ys), bs[0]


@torch.no_grad()
def diagnose_controls(model, device, epoch):
    diag_path = Path(r"G:\Data\color\tests\035_hdlp_015_ORIGINAL.png")
    if not diag_path.exists():
        
        return
    model.eval()
    tf  = T.ToTensor()
    img = Image.open(diag_path).convert("RGB")
    res = CONF["res"];  w,h = img.size
    if w>=h: new_w=res; new_h=max(8,int(round(h*(res/w))))
    else:    new_h=res; new_w=max(8,int(round(w*(res/h))))
    new_w=max(8,(new_w//8)*8);  new_h=max(8,(new_h//8)*8)
    img_rs = img.resize((new_w,new_h),Image.LANCZOS)
    x384   = tf(img_rs).unsqueeze(0).to(device)
    xfull  = tf(img).unsqueeze(0).to(device)
    out, aux = model(xfull, x384)
    hgp      = highlight_gradient_preserve_loss(out,x_orig=xfull,lo=0.80,weight=1.0)
    hi_brake = input_highlight_brake_loss(out,xfull,lo=0.88,weight=1.0)
    ev       = aux["ev"].item()
    hi       = aux["hi"].item()
    shadows  = aux["shadows"].item()
    
    curve    = aux["curve"].view(-1).cpu().numpy()
    linear   = np.linspace(curve[0],curve[-1],len(curve))
    curve_dev= np.abs(curve-linear).max()

    cg = aux["chroma_grid"].cpu()
    cg_abs = cg.abs().mean().item()
    cg_max = cg.abs().max().item()
   

    temp = aux["temp"].item()
    tint = aux["tint"].item()
    base_mired = 1e6 / 6500.0
    cct = 1e6 / (base_mired + temp * 120.0)

    print(f"   Temp={temp:.4f} Tint={tint:.4f} Shadows={shadows:.4f} (CCT≈{cct:.0f}K)")
    print(f"   ChromaGrid mean={cg_abs:.5f} max={cg_max:.5f}")
    print(f"   Curve start={curve[0]:.4f} end={curve[-1]:.4f} dev={curve_dev:.4f}")
    print(f"   HiGradPres={hgp.item():.6f}  HiBrake={hi_brake.item():.6f}")


@torch.no_grad()
def run_inference_on_folder(model, out_dir):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = CONF["paths"]["test_in"];  device = CONF["device"]
    res = CONF["res"];  tf = T.ToTensor()
    files = [p for p in test_dir.glob("*.*")
             if p.suffix.lower() in [".jpg",".jpeg",".png",".webp"]]
    for f in tqdm(files, desc="Inference", leave=False):
        try:
            img = Image.open(f).convert("RGB");  w,h = img.size
            if w>=h: new_w=res; new_h=max(8,int(round(h*(res/w))))
            else:    new_h=res; new_w=max(8,int(round(w*(res/h))))
            new_w=max(8,(new_w//8)*8);  new_h=max(8,(new_h//8)*8)
            img_rs = img.resize((new_w,new_h),Image.LANCZOS)
            x384   = tf(img_rs).unsqueeze(0).to(device)
            xfull  = tf(img).unsqueeze(0).to(device)
            out,_  = model(xfull,x384)
            T.ToPILImage()(out.squeeze(0).float().cpu()).save(
                out_dir/f"{f.stem}_out.png")
        except Exception as e:
            print(f"Err {f.name}: {e}")


def main():
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    out_root = CONF["paths"]["out"];  out_root.mkdir(parents=True, exist_ok=True)
   

    NUM_WORKERS = 4
    train_ds = GlobalBucketDataset(CONF["paths"]["processed_train"], aug=True)
    val_ds   = GlobalBucketDataset(CONF["paths"]["processed_val"],   aug=False)
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")

    train_sampler = BucketBatchSampler(train_ds.bucket_indices,
                                       CONF["batch"],shuffle=True,seed=42)
    val_sampler   = BucketBatchSampler(val_ds.bucket_indices,
                                       CONF["batch"],shuffle=False,seed=42)

    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=(NUM_WORKERS>0),
                     prefetch_factor=2 if NUM_WORKERS>0 else None,
                     collate_fn=bucket_collate)
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, **dl_kwargs)
    val_dl   = DataLoader(val_ds,   batch_sampler=val_sampler,   **dl_kwargs)

    device = CONF["device"]
    model  = BilateralGridEditor(
        backbone_name='mobilenetv4_conv_small.e2400_r224_in1k', grid_d=24
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(),
                             lr=CONF["lr"], weight_decay=CONF["wd"])
    WARMUP = 3
    sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[
        torch.optim.lr_scheduler.LinearLR(opt,0.01,1.0,total_iters=WARMUP),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,T_max=CONF["epoch"]-WARMUP,eta_min=1e-6),
    ], milestones=[WARMUP])

    criterion = ColorEnhancementLosses(device=device).to(device)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    best_score = float("inf");  bad_epochs = 0

    for ep in range(1, CONF["epoch"]+1):
        model.train()
        for x, y, _bid in tqdm(train_dl, desc=f"Ep {ep}", leave=True):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).clamp(0.004, 1.0)

            if model.training:
                x = x + (torch.rand_like(x)-0.5)/255.0

            original_input = x.clone()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                pred, aux  = model(x, x)
                need       = correction_need_score(x, y)
                gate = (0.35 + 0.65 * need).clamp(0.35, 1.0)
                loss, _    = criterion(
                    pred, y, x, aux, gate,
                    original_input=original_input,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt);  scaler.update()

        sch.step()

        model.eval()
        v_rgb=v_ssim=v_luma=v_uv=v_detail=v_uven=0.0;  v_n=0
        with torch.no_grad():
            for x, y, _bid in val_dl:
                x = x.to(device,non_blocking=True)
                y = y.to(device,non_blocking=True)
                pred,aux = model(x,x)
                lp = 0.2126*pred[:,0:1]+0.7152*pred[:,1:2]+0.0722*pred[:,2:3]
                lt = 0.2126*y[:,0:1]   +0.7152*y[:,1:2]   +0.0722*y[:,2:3]
                v_ssim  += (1.0-kornia.metrics.ssim(lp,lt,window_size=11).mean()).item()
                gate_ones = torch.ones(x.shape[0],device=device)
                _,ld = criterion(pred,y,x,aux,gate_ones,
                                 original_input=x)
                v_rgb    += ld["rgb"].item()
                v_luma   += ld["luma_match"].item()
                v_uv     += ld["yuv_match"].item()
                v_detail += ld["detail_preserve"].item()
                v_uven   += ld["uv_energy"].item()
                v_n      += 1

        n = max(v_n,1)
        avg_rgb=v_rgb/n; avg_ssim=v_ssim/n; avg_luma=v_luma/n
        avg_uv=v_uv/n;   avg_detail=v_detail/n; avg_uven=v_uven/n

        diagnose_controls(model, device, ep)

        score = (avg_rgb + 0.30*avg_ssim + 0.35*avg_luma +
                 0.25*avg_uv + 0.20*avg_detail + 0.15*avg_uven)

        

        if ep >= CONF["min_epoch_to_save"]:
            if score < best_score - 1e-4:
                best_score = score;  bad_epochs = 0
                save_dir = out_root/f"best_ep{ep}_sc{best_score:.5f}"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir/"model.pth")
                with open(save_dir/"info.txt","w") as f:
                    f.write("BASE_LOSS_WEIGHTS = {\n")
                    for k,v in BASE_LOSS_WEIGHTS.items():
                        f.write(f"    {repr(k)}: {v},\n")
                    f.write("}\n")
                run_inference_on_folder(model, save_dir)
               
            else:
                bad_epochs += 1
                if bad_epochs >= CONF["patience"]:
                   
                    break

if __name__ == "__main__":
    main()