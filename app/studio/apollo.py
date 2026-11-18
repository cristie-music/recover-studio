"""Apollo: band-sequence modelling for high-quality music restoration (Kai Li, Yi Luo; Tsinghua / Tencent AI Lab).
https://github.com/JusperLee/Apollo  (code and weights: CC BY-SA 4.0). Pure PyTorch port of look2hear/models/apollo.py:
same layers and parameter names so the original checkpoint loads; hyper-parameters are read from the weights.

The model regenerates the mid/high band of damaged music (codec artefacts, missing highs). It is applied to the mid
and side channels separately (as the original model processes each channel independently), in overlapping windows.
"""
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import device as D
from . import models as M


class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, x):
        B, N, T = x.shape
        xf = x.reshape(B, self.groups, -1, T).float()
        xn = xf * torch.rsqrt(xf.pow(2).mean(-2, keepdim=True) + self.eps)
        return xn.type_as(x).reshape(B, N, T) * self.weight.reshape(1, -1, 1)


class Roformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_head=8, theta=10000, window=10000):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        freq = 1. / (theta ** (torch.arange(0, self.hidden_size, 2)[:(self.hidden_size // 2)] / self.hidden_size))
        pos = torch.arange(0, window).reshape(-1, 1)
        cos_freq = torch.stack([torch.cos(pos * freq.reshape(1, -1))] * 2, -1).reshape(window, self.hidden_size)
        sin_freq = torch.stack([torch.sin(pos * freq.reshape(1, -1))] * 2, -1).reshape(window, self.hidden_size)
        self.register_buffer("cos_freq", cos_freq)
        self.register_buffer("sin_freq", sin_freq)
        self.input_norm = RMSNorm(self.input_size)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size * self.num_head * 3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size * self.num_head, self.input_size, 1, bias=False)
        self.MLP = nn.Sequential(RMSNorm(self.input_size), nn.Conv1d(self.input_size, self.input_size * 8, 1, bias=False),
                                 nn.SiLU())
        self.MLP_output = nn.Conv1d(self.input_size * 4, self.input_size, 1, bias=False)

    def _rope(self, feature):
        T, N = feature.shape[-2:]
        fr = feature.reshape(-1, T, N)
        cos, sin = self.cos_freq[:T].to(feature.dtype), self.sin_freq[:T].to(feature.dtype)
        sign = torch.tensor([-1, 1], device=feature.device, dtype=feature.dtype)
        neg = (torch.flip(fr.reshape(-1, N // 2, 2), [-1]) * sign.reshape(1, 1, 2)).reshape(-1, T, N)
        return (fr * cos.unsqueeze(0) + neg * sin.unsqueeze(0)).reshape(feature.shape)

    def forward(self, x):
        B, _, T = x.shape
        w = self.weight(self.input_norm(x)).reshape(B, self.num_head, self.hidden_size * 3, T).mT
        Q, K, V = torch.split(w, self.hidden_size, dim=-1)
        att = F.scaled_dot_product_attention(self._rope(Q).contiguous(), self._rope(K).contiguous(), V.contiguous())
        out = self.output(att.mT.reshape(B, -1, T)) + x
        gate, z = self.MLP(out).chunk(2, dim=1)
        return out + self.MLP_output(F.silu(gate) * z)


class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel - 1) // 2, groups=in_channel),
                                  RMSNorm(in_channel), nn.Conv1d(in_channel, hidden_channel, 1), nn.SiLU(),
                                  nn.Conv1d(hidden_channel, in_channel, 1))

    def forward(self, x):
        return x + self.conv(x)


class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvActNorm1d(in_channel, in_channel * 4, kernel) for _ in range(3)])

    def forward(self, x):
        return self.blocks(x)


class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super().__init__()
        self.band_net = Roformer(feature_dim, feature_dim, num_head=8, window=100)
        self.seq_net = ICB(feature_dim, kernel=kernel)

    def forward(self, x):
        B, nband, N, T = x.shape
        b = self.band_net(x.permute(0, 3, 2, 1).reshape(B * T, -1, nband))
        b = b.reshape(B, T, -1, nband).permute(0, 3, 2, 1)
        return self.seq_net(b.reshape(B * nband, -1, T)).reshape(B, nband, -1, T)


class Apollo(nn.Module):
    def __init__(self, win=882, feature_dim=256, layer=6):
        super().__init__()
        self.win = int(win)                     # 20 ms at 44.1 kHz in the released model
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.eps = torch.finfo(torch.float32).eps
        bw = int(self.win / 160)
        self.band_width = [bw] * 79 + [self.enc_dim - bw * 79]
        self.nband = len(self.band_width)
        self.BN = nn.ModuleList([nn.Sequential(RMSNorm(w * 2 + 1), nn.Conv1d(w * 2 + 1, feature_dim, 1))
                                 for w in self.band_width])
        self.net = nn.Sequential(*[BSNet(feature_dim) for _ in range(layer)])
        self.output = nn.ModuleList([nn.Sequential(RMSNorm(feature_dim), nn.Conv1d(feature_dim, w * 4, 1), nn.GLU(dim=1))
                                     for w in self.band_width])

    def forward(self, x):
        """x (B, ch, n) -> (B, ch, n); every channel is processed independently."""
        B, nch, n = x.shape
        win = torch.hann_window(self.win, device=x.device)
        spec = torch.stft(x.reshape(B * nch, n).float(), n_fft=self.win, hop_length=self.stride, window=win,
                          return_complex=True)
        feats, i = [], 0
        for k, w in enumerate(self.band_width):
            s = spec[:, i:i + w]
            p = (s.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1)
            sn = s / p
            f = torch.cat([sn.real, sn.imag, torch.log(p)], 1).to(x.dtype if x.dtype != torch.float32 else torch.float32)
            feats.append(self.BN[k](f))
            i += w
        feat = self.net(torch.stack(feats, 1))
        est = []
        for k, w in enumerate(self.band_width):
            ri = self.output[k](feat[:, k]).float().view(B * nch, 2, w, -1)
            est.append(torch.complex(ri[:, 0], ri[:, 1]))
        y = torch.istft(torch.cat(est, 1), n_fft=self.win, hop_length=self.stride, window=win, length=n)
        return y.view(B, nch, -1)


_NET = {}


class _Stub:
    """Placeholder for classes pickled next to the weights (omegaconf configs etc.) - they are not needed."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *a, **k):
        return _Stub()


class _StubPickle:
    """pickle_module for torch.load: real classes from torch/collections/numpy, stubs for anything else
    (so no omegaconf / pytorch_lightning import and no arbitrary code from the checkpoint)."""
    import pickle as _p
    __name__ = "pickle"
    load = _p.load
    loads = _p.loads

    class Unpickler(_p.Unpickler):
        def find_class(self, module, name):
            if module.split(".")[0] in ("torch", "collections", "numpy", "_codecs", "builtins") and \
                    not (module == "builtins" and name in ("eval", "exec", "getattr", "__import__", "open")):
                return super().find_class(module, name)
            return _Stub


def load(dev):
    if "net" in _NET and _NET["dev"] == str(dev):
        return _NET["net"]
    sd = torch.load(M.path(M.MODELS_INFO["apollo"]["ckpt"]), map_location="cpu", weights_only=False,
                    pickle_module=_StubPickle)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {re.sub(r"^(module\.|audio_model\.|model\.)", "", k): v for k, v in sd.items()}
    feature_dim = sd["BN.0.1.weight"].shape[0]
    bw = (sd["BN.0.1.weight"].shape[1] - 1) // 2
    layers = len({k.split(".")[1] for k in sd if k.startswith("net.")})
    last = (sd["BN.79.1.weight"].shape[1] - 1) // 2
    win = (bw * 79 + last - 1) * 2                   # enc_dim = win // 2 + 1 = 79 equal bands + the rest
    net = Apollo(win, feature_dim, layers)
    missing, unexpected = net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    missing = [k for k in missing if not k.endswith(("cos_freq", "sin_freq"))]
    if missing or [k for k in unexpected if not k.endswith(("cos_freq", "sin_freq"))]:
        raise RuntimeError(f"Apollo: веса не подходят (нет {missing[:3]}, лишние {unexpected[:3]})")
    net.to(dev).eval()
    _NET.update(net=net, dev=str(dev))
    return net


def unload():
    _NET.clear()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def restore(audio, want_device="auto", chunk_s=10.0, overlap=2, progress=None, log=print):
    """audio float32 (2, n) at 44.1 kHz -> restored (2, n). Mid/side are processed as two independent channels.
    The output level is matched to the input per window (Apollo was trained on peak-normalised audio)."""
    dev, dinfo = D.torch_device(want_device)
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    net = load(dev)
    sr = 44100
    chunk = int(chunk_s * sr)
    hop = chunk // overlap
    n = audio.shape[1]
    mid = (audio[0] + audio[1]) / 2
    side = (audio[0] - audio[1]) / 2
    ms = np.stack([mid, side]).astype(np.float32)
    scale = float(np.abs(ms).max()) + 1e-9
    ms = ms / scale
    pad = chunk
    x = np.pad(ms, ((0, 0), (pad, pad + chunk)))
    out = np.zeros_like(x)
    wsum = np.zeros(x.shape[1], np.float32)
    w = np.hanning(chunk).astype(np.float32) + 1e-3
    starts = list(range(0, pad + n + 1, hop))
    batch = 1
    if dev.type == "cuda":
        batch = 4 if (dinfo.get("vram_gb") or 4) >= 14 else 2 if (dinfo.get("vram_gb") or 4) >= 7 else 1
    amp = dev.type == "cuda"
    i, t0 = 0, time.time()
    with torch.inference_mode():
        while i < len(starts):
            ss = starts[i:i + batch]
            try:
                xb = torch.from_numpy(np.stack([x[:, s:s + chunk] for s in ss])).to(dev)
                if amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        yb = net(xb).float().cpu().numpy()
                else:
                    yb = net(xb).float().cpu().numpy()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if dev.type == "cpu":
                    raise
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if batch > 1 and "out of memory" in str(e).lower():
                    batch //= 2
                    log(f"  ! мало видеопамяти -> пакет {batch}")
                else:
                    log(f"  ! ускоритель не справился ({str(e)[:120]}) -> продолжаю на CPU")
                    dev, amp = torch.device("cpu"), False
                    net.to(dev)
                    _NET.clear()
                continue
            for s, y in zip(ss, yb):
                if not np.all(np.isfinite(y)):
                    y = x[:, s:s + chunk]
                out[:, s:s + chunk] += y * w
                wsum[s:s + chunk] += w
            i += len(ss)
            if progress:
                el = time.time() - t0
                progress(i / len(starts), f"{i}/{len(starts)} окон, осталось ~{el / i * (len(starts) - i):.0f} с")
    out = out / np.maximum(wsum, 1e-6)
    out = out[:, pad:pad + n] * scale
    m, s = out[0], out[1]
    return np.stack([m + s, m - s]).astype(np.float32), dict(device=D.label(dinfo), seconds=round(time.time() - t0, 1))
