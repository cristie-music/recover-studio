"""Audio file helpers: read anything libsndfile knows (+ audioread/ffmpeg fallback), resample, write WAV/MP3,
waveform peaks, spectrogram thumbnails, loudness."""
import io
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 44100
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".oga", ".aif", ".aiff", ".m4a", ".aac", ".opus", ".wma")


def read_any(path):
    """-> (float32 array (n, ch), sr)."""
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except Exception as e1:
        data = open(path, "rb").read()
        i = data.find(b"OggS")
        if 0 <= i < 4096:
            return sf.read(io.BytesIO(data[i:]), dtype="float32", always_2d=True)
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=False)
            y = np.atleast_2d(y).T.astype(np.float32)
            return y, sr
        except Exception:
            raise RuntimeError(f"не удалось прочитать {os.path.basename(path)}: {e1}. "
                               "Поддерживаются WAV, FLAC, MP3, OGG, AIFF (для M4A/AAC нужен ffmpeg).")


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def load_stereo(path, sr=SR):
    """-> (float32 (n, 2) at sr, original sr, original channel count)."""
    x, sr_in = read_any(path)
    ch = x.shape[1]
    if ch == 1:
        x = np.repeat(x, 2, 1)
    x = resample(x[:, :2], sr_in, sr)
    return np.ascontiguousarray(x, dtype=np.float32), sr_in, ch


BITS = {"16": "PCM_16", "24": "PCM_24", "32": "FLOAT"}


def write_wav(path, x, sr=SR, bits="24"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    subtype = BITS.get(str(bits), "PCM_24")
    y = x
    if subtype == "PCM_16":   # TPDF dither for 16 bit
        y = x + (np.random.rand(*x.shape) - np.random.rand(*x.shape)).astype(np.float32) / 32768.0
    if subtype != "FLOAT":
        y = np.clip(y, -1.0, 1.0 - 1e-6)
    sf.write(path + ".tmp.wav", y, sr, subtype=subtype)
    os.replace(path + ".tmp.wav", path)


def write_mp3(path, x, sr=SR):
    """Small preview for the browser. Written in blocks (libsndfile's encoders crash on one huge write)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    y = np.clip(x, -1, 1).astype(np.float32)
    if y.ndim == 1:
        y = y[:, None]
    try:
        with sf.SoundFile(path + ".tmp", "w", sr, y.shape[1], format="MP3", subtype="MPEG_LAYER_III") as f:
            for i in range(0, len(y), 16384):
                f.write(y[i:i + 16384])
        os.replace(path + ".tmp", path)
        return path
    except Exception:   # old libsndfile without MP3: OGG preview instead
        p2 = os.path.splitext(path)[0] + ".ogg"
        with sf.SoundFile(p2, "w", sr, y.shape[1], format="OGG", subtype="VORBIS") as f:
            for i in range(0, len(y), 16384):
                f.write(y[i:i + 16384])
        return p2


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def loudness(x, sr=SR):
    """Integrated loudness, LUFS (ITU-R BS.1770)."""
    import pyloudnorm as pyln
    if len(x) < sr * 0.5:
        return float("-inf")
    try:
        return float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
    except Exception:
        return float("-inf")


def true_peak_db(x, sr=SR):
    """4x oversampled peak, dBTP (computed in blocks to keep memory low)."""
    pk = 0.0
    blk = sr * 10
    for i in range(0, len(x), blk):
        seg = x[max(0, i - 64): i + blk + 64]
        pk = max(pk, float(np.abs(resample_poly(seg, 4, 1, axis=0)).max()))
    return float(20 * np.log10(pk + 1e-12))


def peaks(x, points=1600):
    """Waveform overview: max |amplitude| per column (0..1), mono."""
    m = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    n = max(1, len(m) // points)
    m = m[: n * (len(m) // n)].reshape(-1, n).max(1)
    return [round(float(v), 3) for v in m]


def peaks_cached(path, points=1600):
    cache = path + ".peaks.json"
    try:
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
            d = json.load(open(cache))
            if len(d.get("peaks", [])) >= min(points, 200):
                return d
    except (OSError, ValueError):
        pass
    x, sr = read_any(path)
    d = dict(peaks=peaks(x, points), duration=len(x) / sr)
    try:
        json.dump(d, open(cache, "w"))
    except OSError:
        pass
    return d


def spectrogram(x, sr=SR, cols=900, rows=160, fmax=22050.0):
    """Log-frequency spectrogram thumbnail for the browser: uint8 rows x cols (0 = -100 dB, 255 = 0 dB),
    frequency axis 30 Hz .. fmax on a log scale (low rows = low freqs)."""
    m = x.mean(1) if x.ndim == 2 else x
    n_fft = 4096
    hop = max(256, len(m) // cols)
    nfr = max(1, (len(m) - n_fft) // hop + 1)
    win = np.hanning(n_fft).astype(np.float32)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(30, min(fmax, sr / 2), rows + 1)
    idx = np.clip(np.searchsorted(f, edges), 1, len(f) - 1)
    out = np.zeros((rows, nfr), np.float32)
    for s in range(0, nfr, 256):
        e = min(nfr, s + 256)
        fr = np.stack([m[i * hop: i * hop + n_fft] for i in range(s, e)]) * win
        P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 / (win.sum() ** 2 / 4)
        c = np.concatenate([np.zeros((P.shape[0], 1)), np.cumsum(P, 1)], 1)
        a, b = idx[:-1], np.maximum(idx[1:], idx[:-1] + 1)
        band = (c[:, b] - c[:, a]) / (b - a)
        out[:, s:e] = band.T
    db = 10 * np.log10(out + 1e-12)
    img = np.clip((db + 100) / 100 * 255, 0, 255).astype(np.uint8)
    return dict(rows=rows, cols=nfr, fmin=30, fmax=float(edges[-1]), data=img.tolist())
