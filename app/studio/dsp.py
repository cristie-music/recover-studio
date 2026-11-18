"""Classic restoration DSP (CPU, numpy/scipy). Every stage works on float32 (n, 2) arrays at 44.1 kHz.

  rumble / DC        high-pass (zero phase)
  de-hum             mains hum 50/60 Hz + harmonics: detected, removed with narrow zero-phase notches
  de-click/crackle   AR-residual detection + least-squares AR interpolation (LSAR, Godsill & Rayner)
  de-clip            clipped runs re-drawn with LSAR, never below the clip level
  de-wow / flutter   speed drift tracked from the log-frequency spectrum, removed by time-varying resampling
  denoise            spectral (Wiener) noise reduction with a learned noise profile (CPU fallback for the AI denoiser)
  tonal rebalance    long-term spectrum matched to a target curve or a reference track (linear-phase FIR)
  bandwidth ext.     harmonic exciter regenerates the band above the detected cut-off
  stereo             mid/side width
  master             glue compressor, tape saturation, true-peak limiter, LUFS target
Parts adapted from Vocal Cleaner (declick masks, limiter) by the same author.
"""
import numpy as np
import scipy.linalg as sl
import scipy.signal as ss
from scipy.ndimage import binary_dilation, label, maximum_filter1d, minimum_filter1d, uniform_filter1d

SR = 44100


def db(v):
    return 20 * np.log10(np.maximum(np.abs(v), 1e-12))


def undb(d):
    return np.power(10.0, np.asarray(d) / 20.0)


def frame_rms_db(m, sr=SR, step=0.01, win=None):
    hop = int(sr * step)
    frames = len(m) // hop
    p = (m[: frames * hop].reshape(frames, hop) ** 2).mean(1)
    if win:
        p = uniform_filter1d(p, max(1, int(win / step)))
    return 10 * np.log10(p + 1e-12)


def active_level(x, sr=SR):
    """Level (dBFS RMS) of the loud part of the signal: mean of 50 ms frames within 20 dB of the 95th percentile."""
    m = x.mean(1) if x.ndim == 2 else x
    lv = frame_rms_db(m, sr, 0.05)
    if not len(lv):
        return -100.0
    top = np.percentile(lv, 95)
    sel = lv[lv > top - 20]
    return float(10 * np.log10(np.mean(10 ** (sel / 10)) + 1e-12))


# ================================================================ filters
def highpass(x, sr, f, order=4, zero_phase=True):
    if not f or f <= 0:
        return x
    sos = ss.butter(order, f, "hp", fs=sr, output="sos")
    return (ss.sosfiltfilt(sos, x, axis=0) if zero_phase else ss.sosfilt(sos, x, axis=0)).astype(np.float32)


def lowpass(x, sr, f, order=4, zero_phase=True):
    if not f or f >= 0.49 * sr:
        return x
    sos = ss.butter(order, f, "lp", fs=sr, output="sos")
    return (ss.sosfiltfilt(sos, x, axis=0) if zero_phase else ss.sosfilt(sos, x, axis=0)).astype(np.float32)


def _biquad(kind, f, g_db, q, sr):
    """RBJ cookbook biquad -> sos row."""
    A = 10 ** (g_db / 40)
    w = 2 * np.pi * f / sr
    c, s = np.cos(w), np.sin(w)
    al = s / (2 * q)
    if kind == "bell":
        b = [1 + al * A, -2 * c, 1 - al * A]
        a = [1 + al / A, -2 * c, 1 - al / A]
    elif kind == "low_shelf":
        sq = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)]
        a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq]
    else:  # high_shelf
        sq = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)]
        a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq]
    b, a = np.array(b) / a[0], np.array(a) / a[0]
    return np.concatenate([b, a])


def eq(x, sr, bands):
    """bands: [(kind, freq, gain_db, q)], kind = bell | low_shelf | high_shelf."""
    rows = [_biquad(k, f, g, q, sr) for k, f, g, q in bands if g and f < 0.48 * sr]
    if not rows:
        return x
    return ss.sosfilt(np.array(rows), x, axis=0).astype(np.float32)


# ================================================================ spectrum helpers
def ltas(x, sr=SR, n_fft=8192, max_frames=4000, gate_db=-45.0):
    """Long-term average spectrum (power, mono) over the frames that are within gate_db of the loud level.
    -> (freqs, power, noise_power) where noise is the 8th percentile of each bin (stationary noise estimate)."""
    m = x.mean(1) if x.ndim == 2 else x
    hop = n_fft // 2
    nfr = max(1, (len(m) - n_fft) // hop + 1)
    step = max(1, nfr // max_frames)
    win = np.hanning(n_fft).astype(np.float32)
    idx = np.arange(0, nfr, step)
    P = []
    for s in range(0, len(idx), 512):
        fr = np.stack([m[i * hop: i * hop + n_fft] for i in idx[s:s + 512]]) * win
        P.append(np.abs(np.fft.rfft(fr, axis=1)) ** 2 / (win.sum() ** 2 / 4))
    P = np.concatenate(P)
    lv = 10 * np.log10(P.sum(1) + 1e-20)
    live = lv > np.percentile(lv, 95) + gate_db
    if live.sum() < 4:
        live[:] = True
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    return f, P[live].mean(0), np.percentile(P, 8, axis=0)


def octave_smooth(f, p_db, frac=3.0):
    """Smooth a dB spectrum over 1/frac octave (on linear power)."""
    lin = 10 ** (p_db / 10)
    c = np.concatenate([[0], np.cumsum(lin)])
    lo = np.searchsorted(f, f * 2 ** (-0.5 / frac))
    hi = np.maximum(np.searchsorted(f, f * 2 ** (0.5 / frac)), lo + 1)
    hi = np.minimum(hi, len(f))
    return 10 * np.log10((c[hi] - c[lo]) / (hi - lo) + 1e-20)


def band_curve(x, sr=SR, freqs=None):
    """1/3-octave smoothed LTAS in dB on a fixed log grid (for matching and the UI)."""
    if freqs is None:
        freqs = GRID
    f, P, N = ltas(x, sr)
    s = octave_smooth(f, 10 * np.log10(P + 1e-20), 3)
    n = octave_smooth(f, 10 * np.log10(N + 1e-20), 3)
    return np.interp(freqs, f, s), np.interp(freqs, f, n)


GRID = np.geomspace(20, 20000, 61)


def norm_curve(c, freqs=None):
    """Curve normalised to 0 dB as the power-mean of 200 Hz - 4 kHz."""
    freqs = GRID if freqs is None else freqs
    sel = (freqs >= 200) & (freqs <= 4000)
    return c - 10 * np.log10(np.mean(10 ** (c[sel] / 10)))


def detect_cutoff(x, sr=SR):
    """Upper bandwidth of the recording: highest frequency where the 1/6-oct LTAS is within 35 dB of the
    1-3 kHz level, followed by a steep drop. Returns Hz (sr/2 if full band)."""
    f, P, _ = ltas(x, sr)
    s = octave_smooth(f, 10 * np.log10(P + 1e-20), 6)
    ref = np.mean(s[(f > 1000) & (f < 3000)])
    ok = np.where((s > ref - 38) & (f > 1500))[0]
    if not len(ok):
        return 4000.0
    fc = float(f[ok.max()])
    return fc if fc < 0.45 * sr else sr / 2


def detect_lowcut(x, sr=SR):
    """Lower bandwidth: lowest frequency where the 1/3-oct LTAS is within 25 dB of the 200-1000 Hz level."""
    f, P, _ = ltas(x, sr)
    s = octave_smooth(f, 10 * np.log10(P + 1e-20), 3)
    ref = np.mean(s[(f > 200) & (f < 1000)])
    ok = np.where((s > ref - 25) & (f > 15))[0]
    return float(f[ok.min()]) if len(ok) else 200.0


# ================================================================ hum
def detect_hum(x, sr=SR, seconds=90):
    """-> dict(f0=50|60|None, freq=exact Hz, harmonics=[(n, prominence dB)], level_db)."""
    m = x.mean(1)[: int(seconds * sr)] if x.ndim == 2 else x[: int(seconds * sr)]
    f, P = ss.welch(m, sr, nperseg=int(sr * 4), noverlap=int(sr * 2))   # 0.25 Hz bins
    Pd = 10 * np.log10(P + 1e-20)
    df = f[1] - f[0]
    best = None
    for f0 in (50.0, 60.0):
        harm = []
        score = 0.0
        exact = f0
        for n in range(1, 13):
            fc = f0 * n if n == 1 else exact * n
            if fc > 1500:
                break
            i0, i1 = np.searchsorted(f, fc - 1.2 * (1 if n == 1 else 0.3 * n ** 0.5)), np.searchsorted(f, fc + 1.2 * (1 if n == 1 else 0.3 * n ** 0.5))
            if i1 <= i0:
                continue
            k = i0 + int(np.argmax(Pd[i0:i1 + 1]))
            nb = np.r_[max(0, k - int(8 / df)):max(0, k - int(2 / df)), k + int(2 / df):k + int(8 / df)]
            prom = float(Pd[k] - np.median(Pd[nb]))
            if n == 1 and prom > 6:
                exact = float(f[k])
            if prom > 6:
                harm.append((n, round(prom, 1)))
                score += prom * (1.5 if n <= 3 else 1.0)
        if best is None or score > best["score"]:
            best = dict(f0=f0, freq=exact if harm and harm[0][0] == 1 else f0, harmonics=harm, score=score)
    if not best or best["score"] < 18 or len(best["harmonics"]) < 2:
        return dict(f0=None, freq=None, harmonics=[], score=round(best["score"] if best else 0, 1))
    best["score"] = round(best["score"], 1)
    return best


def dehum(x, sr, hum, depth=1.0, max_harm=12):
    """Zero-phase notches on every detected harmonic (+ the odd/even neighbours, which hum always has a bit of).
    depth 0..1 blends between no and full removal."""
    if not hum or not hum.get("f0") or depth <= 0:
        return x, 0
    f0 = hum["freq"] or hum["f0"]
    found = {n for n, _ in hum["harmonics"]}
    top = max(found) if found else 3
    y = x.astype(np.float64)
    count = 0
    for n in range(1, min(max_harm, top + 2) + 1):
        fc = f0 * n
        if fc > 0.45 * sr:
            break
        bw = 1.2 + 0.25 * n                      # Hz; harmonics wander a bit more
        b, a = ss.iirnotch(fc, fc / bw, sr)
        y = ss.filtfilt(b, a, y, axis=0)
        count += 1
    out = (x * (1 - depth) + y * depth).astype(np.float32)
    return out, count


# ================================================================ clicks / crackle (LSAR)
def _lpc(seg, order):
    r = np.correlate(seg, seg, "full")[len(seg) - 1: len(seg) + order]
    if r[0] <= 1e-12:
        return None
    r = r.astype(np.float64)
    r[0] *= 1.0 + 1e-6
    try:
        a = sl.solve_toeplitz(r[:-1], -r[1:])
    except Exception:
        return None
    return np.concatenate([[1.0], a])


def _lsar(seg, miss, a):
    """Least-squares AR interpolation: unknown samples (bool mask) that minimise the prediction error."""
    p = len(a) - 1
    n = len(seg)
    rows = n - p
    if rows <= 0:
        return seg
    # prediction-error matrix A (rows x n): e[t] = sum_k a[k] x[t+p-k]
    A = np.zeros((rows, n))
    for k in range(p + 1):
        A[np.arange(rows), np.arange(rows) + p - k] = a[k]
    Au, Ak = A[:, miss], A[:, ~miss]
    try:
        xu = -np.linalg.lstsq(Au, Ak @ seg[~miss], rcond=None)[0]
    except np.linalg.LinAlgError:
        return seg
    out = seg.copy()
    out[miss] = xu
    return out


def _residual(m, sr, order=24, block=4096):
    """AR prediction residual of a mono signal, blockwise LPC (robust scale per block)."""
    e = np.zeros_like(m, dtype=np.float64)
    sig = np.zeros(len(m) // block + 1)
    for bi, s in enumerate(range(0, len(m), block)):
        seg = m[max(0, s - order): s + block].astype(np.float64)
        a = _lpc(seg * np.hanning(len(seg)), order) if len(seg) > 4 * order else None
        if a is None:
            continue
        r = ss.lfilter(a, [1.0], seg)[min(order, s):]
        e[s:s + len(r)] = r[: len(e) - s]
        sig[bi] = 1.4826 * np.median(np.abs(r)) + 1e-9
    return e, sig, block


def declick(x, sr, sensitivity=5.0, max_ms=5.0, crackle=False, progress=None):
    """Removes clicks (crackle=True: the dense fine crackle of worn vinyl / shellac).
    Detection: AR residual far above its robust local scale AND isolated (a drum attack has a long residual burst,
    a click a short one). Repair: LSAR interpolation of the damaged samples in every channel.
    sensitivity 1..10. Returns (y, number of events)."""
    if sensitivity <= 0:
        return x, 0
    m = x.mean(1).astype(np.float64)
    order = 24 if not crackle else 16
    e, sig, block = _residual(m, sr, order)
    scale = np.repeat(sig, block)[: len(m)]
    k = (10.5 - 0.6 * sensitivity) if not crackle else (7.0 - 0.4 * sensitivity)
    ratio = np.abs(e) / scale
    hits = ratio > k
    if not hits.any():
        return x, 0
    # isolation test: residual energy around the hit (excluding +-0.3 ms) must be low
    w_iso = int(0.004 * sr)
    ex = int(0.0005 * sr) + 1
    loud = uniform_filter1d((ratio > k * 0.45).astype(np.float32), 2 * w_iso + 1)
    core = uniform_filter1d(binary_dilation(hits, np.ones(2 * ex + 1, bool)).astype(np.float32), 2 * w_iso + 1)
    isolated = (loud - core) < (0.08 if not crackle else 0.10)
    hits &= isolated
    max_len = int(max_ms / 1000 * sr) if not crackle else int(0.0008 * sr) + 2
    pad = 2 if crackle else 6
    hits = binary_dilation(hits, np.ones(2 * pad + 1, bool))
    lab, cnt = label(hits)
    if cnt == 0:
        return x, 0
    objs = __import__("scipy.ndimage", fromlist=["find_objects"]).find_objects(lab)
    y = x.astype(np.float64).copy()
    p = 32 if not crackle else 20
    fixed = 0
    for j, sl_ in enumerate(objs):
        a0, a1 = sl_[0].start, sl_[0].stop
        L = a1 - a0
        if L > max_len:
            continue
        ctx = max(3 * p, 4 * L)
        s0, s1 = a0 - ctx, a1 + ctx
        if s0 < 0 or s1 > len(m):
            continue
        miss = np.zeros(s1 - s0, bool)
        miss[a0 - s0:a1 - s0] = True
        for c in range(y.shape[1]):
            seg = y[s0:s1, c]
            known = np.concatenate([seg[: a0 - s0], seg[a1 - s0:]])
            a = _lpc(known * np.hanning(len(known)), p)
            if a is None:
                continue
            y[s0:s1, c] = _lsar(seg, miss, a)
        fixed += 1
        if progress and j % 500 == 0:
            progress(j / len(objs))
    return y.astype(np.float32), fixed


def declip(x, sr, thresh=0.985):
    """Clipped runs (>= 3 samples at the channel's max) re-drawn by LSAR; the result keeps the sign and never
    goes below the clip level. Returns (y, runs fixed). The output may exceed 0 dBFS - the master stage handles it."""
    y = x.astype(np.float64).copy()
    fixed = 0
    for c in range(x.shape[1]):
        v = y[:, c]
        peak = np.abs(v).max()
        if peak <= 0:
            continue
        clip = np.abs(v) >= thresh * peak
        lab, cnt = label(clip)
        if cnt == 0 or cnt > 200000:
            continue
        objs = __import__("scipy.ndimage", fromlist=["find_objects"]).find_objects(lab)
        for sl_ in objs:
            a0, a1 = sl_[0].start, sl_[0].stop
            L = a1 - a0
            if L < 3 or L > int(0.004 * sr):
                continue
            ctx = max(96, 4 * L)
            s0, s1 = a0 - ctx, a1 + ctx
            if s0 < 0 or s1 > len(v):
                continue
            seg = v[s0:s1]
            miss = np.zeros(len(seg), bool)
            miss[a0 - s0:a1 - s0] = True
            known = seg[~miss]
            a = _lpc(known * np.hanning(len(known)), 24)
            if a is None:
                continue
            r = _lsar(seg, miss, a)
            sgn = np.sign(seg[miss])
            r[miss] = sgn * np.maximum(np.abs(r[miss]), np.abs(seg[miss]))
            r[miss] = sgn * np.minimum(np.abs(r[miss]), 2.0 * thresh * peak)   # keep it sane
            v[s0:s1] = r
            fixed += 1
        y[:, c] = v
    return y.astype(np.float32), fixed


def clip_ratio(x):
    peak = np.abs(x).max() + 1e-12
    c = (np.abs(x) >= 0.985 * peak)
    # only runs >= 3 samples count
    run = minimum_filter1d(c.astype(np.uint8), 3, axis=0)
    return float(run.mean())


def count_clicks(x, sr, sensitivity=5.0):
    """Clicks per minute (detection only, on the first 60 s)."""
    seg = x[: int(60 * sr)]
    m = seg.mean(1).astype(np.float64)
    e, sig, block = _residual(m, sr, 24)
    scale = np.repeat(sig, block)[: len(m)]
    ratio = np.abs(e) / scale
    k = 10.5 - 0.6 * sensitivity
    hits = ratio > k
    w_iso = int(0.0025 * sr)
    ex = int(0.0003 * sr) + 1
    loud = uniform_filter1d((ratio > k * 0.45).astype(np.float32), 2 * w_iso + 1)
    core = uniform_filter1d(hits.astype(np.float32), 2 * ex + 1)
    hits &= (loud - core * (2 * ex + 1) / (2 * w_iso + 1)) < 0.08
    _, cnt = label(binary_dilation(hits, np.ones(9, bool)))
    return cnt * 60.0 / max(1.0, len(seg) / sr)


# ================================================================ wow & flutter
def track_speed(x, sr=SR, hop_s=0.02, fmin=150.0, fmax=5000.0, ref_s=1.5):
    """Relative speed deviation (cents) over time from the whole-mix spectrum: every frame's log-frequency spectrum
    is cross-correlated with the average of its +-ref_s neighbourhood (a periodic wow averages out there).
    -> (times, cents, confidence)."""
    m = x.mean(1) if x.ndim == 2 else x
    sr2 = 22050
    if sr != sr2:
        m = ss.resample_poly(m, 1, 2) if sr == 44100 else ss.resample_poly(m, sr2, sr)
    n_fft = 8192
    hop = int(hop_s * sr2)
    nfr = max(1, (len(m) - n_fft) // hop + 1)
    f = np.fft.rfftfreq(n_fft, 1 / sr2)
    cents_per_bin = 2.0
    grid = fmin * 2 ** (np.arange(0, 1200 * np.log2(fmax / fmin), cents_per_bin) / 1200)
    win = np.hanning(n_fft).astype(np.float32)
    S = np.zeros((nfr, len(grid)), np.float32)
    for s in range(0, nfr, 256):
        e = min(nfr, s + 256)
        fr = np.stack([m[i * hop: i * hop + n_fft] for i in range(s, e)]) * win
        mag = np.abs(np.fft.rfft(fr, axis=1))
        lg = np.log1p(mag / (mag.mean(1, keepdims=True) + 1e-9))
        S[s:e] = np.stack([np.interp(grid, f, r) for r in lg])
    # whiten along frequency (keep partial structure, drop the tilt)
    S -= uniform_filter1d(S, 60, axis=1)
    k = max(1, int(ref_s / hop_s))
    R = uniform_filter1d(S, 2 * k + 1, axis=0)
    maxlag = int(60 / cents_per_bin)            # +-60 cents
    F1 = np.fft.rfft(S, n=2 * S.shape[1], axis=1)
    F2 = np.fft.rfft(R, n=2 * S.shape[1], axis=1)
    cc = np.fft.irfft(F1 * np.conj(F2), axis=1)
    cc = np.concatenate([cc[:, -maxlag:], cc[:, :maxlag + 1]], 1)
    norm = np.sqrt((S ** 2).sum(1) * (R ** 2).sum(1)) + 1e-9
    j = cc.argmax(1)
    pk = cc[np.arange(len(j)), j] / norm
    jj = np.clip(j, 1, cc.shape[1] - 2)
    y0, y1, y2 = cc[np.arange(len(jj)), jj - 1], cc[np.arange(len(jj)), jj], cc[np.arange(len(jj)), jj + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    cents = ((jj + np.clip(d, -1, 1)) - maxlag) * cents_per_bin
    times = (np.arange(nfr) * hop + n_fft / 2) / sr2
    return times, cents.astype(np.float64), pk


def analyze_wow(x, sr=SR, seconds=120):
    """-> dict(wow_cents (RMS, 0.2-4 Hz), flutter_cents (4-12 Hz), rate_hz, confidence)."""
    seg = x[: int(seconds * sr)]
    t, c, conf = track_speed(seg, sr)
    if len(c) < 200:
        return dict(wow_cents=0.0, flutter_cents=0.0, rate_hz=None, confidence=0.0)
    fs = 1 / (t[1] - t[0])
    good = conf > np.percentile(conf, 30)
    c = np.where(good, c, np.interp(t, t[good], c[good]))
    c = c - uniform_filter1d(c, int(6 * fs))
    wow = ss.sosfiltfilt(ss.butter(2, [0.2, 4], "bp", fs=fs, output="sos"), c)
    fl = ss.sosfiltfilt(ss.butter(2, [4, min(12, fs / 2.2)], "bp", fs=fs, output="sos"), c)
    f, P = ss.welch(wow, fs, nperseg=min(len(wow), int(40 * fs)))
    sel = (f > 0.2) & (f < 4)
    rate = float(f[sel][np.argmax(P[sel])]) if sel.any() else None
    # periodicity: peak share of the band power (a real turntable/capstan wow is periodic, music is not)
    per = float(P[sel].max() / (P[sel].sum() + 1e-12)) if sel.any() else 0.0
    return dict(wow_cents=round(float(np.sqrt(np.mean(wow ** 2))), 2), flutter_cents=round(float(np.sqrt(np.mean(fl ** 2))), 2),
                rate_hz=round(rate, 2) if rate else None, periodicity=round(per, 3), confidence=round(float(np.mean(conf)), 3))


def dewow(x, sr, strength=1.0, flutter=True, progress=None):
    """Removes speed drift: tracked deviation (cents) -> smoothed -> time map -> resampling with 4-point
    cubic interpolation. Returns (y, info)."""
    if strength <= 0:
        return x, dict(applied=False)
    t, c, conf = track_speed(x, sr)
    if len(c) < 50:
        return x, dict(applied=False)
    fs = 1 / (t[1] - t[0])
    good = conf > np.percentile(conf, 25)
    c = np.where(good, c, np.interp(t, t[good], c[good]))
    c = c - uniform_filter1d(c, int(8 * fs))          # only the modulation, not the key of the song
    hi = 10.0 if flutter else 4.0
    c = ss.sosfiltfilt(ss.butter(2, min(hi, fs / 2.2), "lp", fs=fs, output="sos"), c)
    c *= strength
    # recorded content runs r = 2^(c/1200) times too fast at t: original time phi(t) = integral of r
    tt = np.arange(len(x)) / sr
    r = 2 ** (np.interp(tt, t, c) / 1200.0)
    phi = np.concatenate([[0.0], np.cumsum(r[:-1]) / sr])
    # output sample k is original time k/sr -> input time phi^-1(k/sr)
    n_out = int(phi[-1] * sr)
    pos = np.interp(np.arange(n_out) / sr, phi, tt) * sr
    y = np.zeros((n_out, x.shape[1]), np.float32)
    i0 = np.floor(pos).astype(np.int64)
    fr = (pos - i0).astype(np.float32)
    xp = np.pad(x, ((2, 3), (0, 0)), mode="edge")
    for s in range(0, n_out, 1 << 20):
        e = min(n_out, s + (1 << 20))
        ii, u = i0[s:e] + 2, fr[s:e][:, None]
        p0, p1, p2, p3 = xp[ii - 1], xp[ii], xp[ii + 1], xp[ii + 2]
        # Catmull-Rom
        y[s:e] = p1 + 0.5 * u * (p2 - p0 + u * (2 * p0 - 5 * p1 + 4 * p2 - p3 + u * (3 * (p1 - p2) + p3 - p0)))
        if progress:
            progress(e / n_out)
    return y, dict(applied=True, rms_cents=round(float(np.sqrt(np.mean(c ** 2))), 2), max_cents=round(float(np.abs(c).max()), 1))


# ================================================================ spectral denoise
def spec_process(x, sr, n, hop, gain_fn):
    """Linked-stereo STFT processing: gain_fn(f, t, P) -> gain (bins x frames), applied to every channel."""
    specs = []
    for c in range(x.shape[1]):
        f, t, S = ss.stft(x[:, c], sr, nperseg=n, noverlap=n - hop)
        specs.append(S)
    P = np.mean([np.abs(S) ** 2 for S in specs], axis=0)
    G = gain_fn(f, t, P).astype(np.float32)
    out = np.zeros_like(x)
    for c, S in enumerate(specs):
        _, y = ss.istft(S * G, sr, nperseg=n, noverlap=n - hop)
        out[:, c] = y[: len(x)]
    return out


def noise_profile(x, sr, n=4096):
    """Noise spectrum (power per bin) from the quietest frames, or None if there is no quiet part."""
    hop = n // 2
    f, t, S = ss.stft(x.mean(1), sr, nperseg=n, noverlap=n - hop)
    P = np.abs(S) ** 2
    lv = 10 * np.log10(P.sum(0) + 1e-20)
    return f, np.percentile(P, 6, axis=1), lv


def denoise(x, sr, reduction_db=12.0, sensitivity=1.0, profile=None):
    """Spectral noise reduction (CPU fallback / gentle polish after the AI denoiser).
    Noise = 6th percentile of every bin (or the profile from a noise-only part); Wiener-style gain with
    over-subtraction `sensitivity`, a floor of -reduction_db, and time/frequency smoothing against musical noise."""
    if reduction_db <= 0:
        return x
    n, hop = 4096, 1024
    floor = undb(-reduction_db) ** 2

    def gains(f, t, P):
        N = profile if profile is not None else np.percentile(P, 6, axis=1)
        N = uniform_filter1d(N, 5)[:, None]
        Ps = uniform_filter1d(P, 3, axis=1)
        snr = np.maximum(Ps / (N + 1e-20) - 1.0, 0)
        g = snr / (snr + sensitivity * 1.5)
        g = np.maximum(g, floor)
        g = uniform_filter1d(uniform_filter1d(g, 3, axis=0), 5, axis=1)
        return np.sqrt(g)

    return spec_process(x, sr, n, hop, gains)


# ================================================================ tonal rebalance
def match_eq(x, sr, target_db, amount=1.0, max_boost=12.0, max_cut=12.0, protect_noise=True, freqs=None,
             lo=25.0, hi=None, numtaps=8191):
    """Linear-phase EQ so that the 1/3-oct long-term spectrum of x follows target_db (dB on `freqs`).
    Boost is limited where the signal is close to its own noise floor (no noise pumping).
    Returns (y, applied_gain_db on freqs)."""
    freqs = GRID if freqs is None else freqs
    cur, noise = band_curve(x, sr, freqs)
    cn = norm_curve(cur, freqs)
    tn = norm_curve(target_db, freqs)
    g = tn - cn
    if protect_noise:
        # typical (median) content vs the hiss / rumble floor per 1/3 octave: a band that is mostly noise is not boosted
        f_ = np.fft.rfftfreq(4096, 1 / sr)
        med = octave_smooth(f_, 10 * np.log10(median_power(x, sr) + 1e-20), 3)
        flo = octave_smooth(f_, 10 * np.log10(noise_floor(x, sr) + 1e-20), 3)
        snr = np.interp(freqs, f_, med - flo)
        lim = np.clip(snr - 8.0, 0, max_boost)           # median less than 8 dB above the floor -> no boost
        g = np.minimum(g, lim)
    g = np.clip(g, -max_cut, max_boost) * amount
    hi = hi or 0.47 * sr
    g[(freqs < lo) | (freqs > hi)] = np.minimum(g[(freqs < lo) | (freqs > hi)], 0)
    # smooth in log-frequency
    g = uniform_filter1d(g, 3, mode="nearest")
    g -= np.median(g[(freqs > 300) & (freqs < 3000)])   # keep the level, change only the balance
    fgrid = np.linspace(0, sr / 2, 2049)
    gl = np.interp(np.log(np.maximum(fgrid, 1)), np.log(freqs), g, left=g[0], right=g[-1])
    h = ss.firwin2(numtaps, fgrid, undb(gl), fs=sr)
    d = numtaps // 2
    y = np.zeros_like(x)
    for c in range(x.shape[1]):
        z = ss.oaconvolve(x[:, c], h)[d:d + len(x)]
        y[:, c] = z
    return y.astype(np.float32), g


# Reference tonal balance: average 1/3-oct LTAS (on GRID, normalised at 200 Hz-4 kHz) of finished rock/pop masters.
# Above 18 kHz the masters are empty (MP3 low-pass), so the top two points are replaced by a gentle slope.
_T = np.array([-14.6, -12.7, -10.1, -6.7, -2.2, 3.0, 7.4, 10.0, 11.2, 12.0, 13.3, 13.7, 12.2, 11.3, 10.7, 10.1, 9.7, 8.2, 6.9, 7.7, 7.7, 5.6, 3.5, 1.7, 1.0, 0.6, 0.5, 0.4, -0.8, -1.8, -1.3, -1.5, -2.6, -2.8, -3.1, -3.7, -5.3, -6.2, -6.5, -5.6, -5.7, -7.5, -9.3, -10.9, -12.1, -12.6, -13.6, -15.6, -17.2, -18.3, -19.5, -20.2, -21.6, -22.8, -24.2, -26.0, -27.8, -29.9, -34.0, -60.7, -81.8])
_T[-2:] = [_T[-3] - 4.0, _T[-3] - 8.0]
TARGET_BALANCED = _T


def style_tilt(freqs, style):
    lf = np.log2(freqs / 1000)
    if style == "warm":
        return np.where(lf < 0, -lf * 0.8, -lf * 0.9)          # +2.7 dB @ 100 Hz, -3 dB @ 10 kHz
    if style == "open":
        return np.where(lf < 0, lf * 0.25, lf * 0.8)            # -0.8 dB @ 100 Hz, +2.7 dB @ 10 kHz
    return np.zeros_like(freqs)


# ================================================================ bandwidth extension
def excite(x, sr, cutoff, amount=1.0, top=19000.0):
    """Regenerates the band above `cutoff` from the octave below it: full-wave/tanh nonlinearity -> harmonics,
    high-passed at the cut-off, envelope follows the source band, level continues the spectral slope.
    amount 0..1.5. Returns (y, gain_db applied to the generated band)."""
    if amount <= 0 or cutoff >= min(top, 0.45 * sr) - 500:
        return x, None
    lo_src = max(800.0, cutoff / 2)
    src = ss.sosfiltfilt(ss.butter(4, [lo_src, cutoff * 0.98], "bandpass", fs=sr, output="sos"), x, axis=0)
    env = np.sqrt(np.maximum(uniform_filter1d(src ** 2, int(0.01 * sr), axis=0), 0)) + 1e-9
    drive = 3.0
    nl = np.tanh(drive * src / (env * 2.5)) * env * 2.5 / drive        # level-independent soft clipping
    nl = nl + 0.5 * (np.abs(src) - uniform_filter1d(np.abs(src), int(0.002 * sr), axis=0))   # even harmonics
    # gate: only where the source band clearly rises above its own noise floor (hiss must not become 'air')
    env_db = 20 * np.log10(env.max(1) + 1e-12)
    nf = np.percentile(env_db, 10)
    gate = np.clip((env_db - (nf + 8.0)) / 12.0, 0, 1)
    gate = uniform_filter1d(gate, int(0.02 * sr)).astype(np.float32)
    nl = nl * gate[:, None]
    gen = ss.sosfiltfilt(ss.butter(6, cutoff, "hp", fs=sr, output="sos"), nl, axis=0)
    gen = ss.sosfiltfilt(ss.butter(4, min(top, 0.47 * sr), "lp", fs=sr, output="sos"), gen, axis=0)
    # level: the generated band should sit ~ (slope) dB below the source octave, about -4.5 dB/oct beyond cutoff
    f, Ps = ss.welch(src.mean(1), sr, nperseg=4096)
    _, Pg = ss.welch(gen.mean(1), sr, nperseg=4096)
    src_lv = 10 * np.log10(Ps[(f > cutoff * 0.7) & (f < cutoff * 0.95)].mean() + 1e-20)
    gen_lv = 10 * np.log10(Pg[(f > cutoff * 1.05) & (f < cutoff * 1.4)].mean() + 1e-20)
    gdb = (src_lv - 6.0) - gen_lv
    gdb = float(np.clip(gdb, -40, 30))
    y = x + gen * undb(gdb) * amount
    return y.astype(np.float32), round(gdb, 1)


# ================================================================ stereo
def stereo_corr(x):
    if x.shape[1] < 2:
        return 1.0
    a, b = x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()
    return float(np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12))


def width(x, w=1.0, bass_mono_hz=120.0, sr=SR):
    """Mid/side width (1 = unchanged); the low end below bass_mono_hz is kept mono when widening."""
    if abs(w - 1.0) < 1e-3:
        return x
    m = (x[:, 0] + x[:, 1]) / 2
    s = (x[:, 0] - x[:, 1]) / 2
    s2 = s * w
    if w > 1 and bass_mono_hz:
        low = ss.sosfiltfilt(ss.butter(2, bass_mono_hz, "lp", fs=sr, output="sos"), s)
        s2 = s2 - low * (w - 1)
    return np.stack([m + s2, m - s2], 1).astype(np.float32)


# ================================================================ dynamics + loudness
def compress(x, sr, ratio=2.0, threshold_rel=-10.0, attack=0.02, release=0.25, knee=6.0, makeup=True):
    """Feed-forward RMS compressor, linked stereo. threshold_rel is relative to the active level."""
    if ratio <= 1.0:
        return x
    act = active_level(x, sr)
    thr = act + threshold_rel
    m = np.max(np.abs(x), 1).astype(np.float64)
    step = 64
    nb = len(m) // step + 1
    pm = np.pad(m ** 2, (0, nb * step - len(m))).reshape(nb, step).mean(1)
    lv = 10 * np.log10(pm + 1e-20) / 1.0
    lv = 10 * np.log10(uniform_filter1d(10 ** (lv / 10), max(1, int(0.01 * sr / step))) + 1e-20)
    over = lv - thr
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    a_c = np.exp(-step / (attack * sr))
    r_c = np.exp(-step / (release * sr))
    g = np.zeros_like(gr)
    prev = 0.0
    for i in range(len(gr)):          # ~70k blocks for 3.5 min: fine in a loop
        v = gr[i]
        prev = a_c * prev + (1 - a_c) * v if v > prev else r_c * prev + (1 - r_c) * v
        g[i] = prev
    gain = undb(-np.interp(np.arange(len(m)), np.arange(nb) * step + step / 2, g))
    y = x * gain[:, None].astype(np.float32)
    if makeup:
        y *= undb(act - active_level(y, sr))
    return y.astype(np.float32), float(np.percentile(g, 95))


def saturate(x, sr, drive_db=3.0, mix=0.5, bias=0.06):
    """Tape-like soft saturation (asymmetric tanh, 2x oversampled), level-matched."""
    if drive_db <= 0 or mix <= 0:
        return x
    act = active_level(x, sr)
    z = x * undb(-18 - act)
    up = ss.resample_poly(z, 2, 1, axis=0)
    g = undb(drive_db)
    y = (np.tanh(g * (up + bias)) - np.tanh(g * bias)) / g
    y = ss.resample_poly(y, 1, 2, axis=0)[: len(z)]
    y = highpass(y, sr, 12, order=2)
    out = z * (1 - mix) + y * mix
    return (out * undb(act - active_level(out, sr))).astype(np.float32)


def limiter(x, sr, ceiling_db=-1.0, release=0.08, look_ms=1.5):
    """Look-ahead true-peak limiter (4x oversampled detection, 1 ms blocks). From Vocal Cleaner."""
    pk = np.zeros(len(x))
    blk = sr * 10
    for i in range(0, len(x), blk):
        seg = x[i:i + blk]
        up = np.abs(ss.resample_poly(seg, 4, 1, axis=0)).max(1)
        up = np.pad(up, (0, (-len(up)) % 4))
        pk[i:i + len(seg)] = up.reshape(-1, 4).max(1)[: len(seg)]
    need = np.minimum(1.0, undb(ceiling_db) / (pk + 1e-12))
    B = max(1, int(sr * 0.001))
    nb = int(np.ceil(len(need) / B))
    padn = np.pad(need, (0, nb * B - len(need)), constant_values=1.0)
    blkm = padn.reshape(nb, B).min(1)
    la = max(1, int(round(look_ms)))
    blkm = minimum_filter1d(blkm, 2 * la + 1)
    rel = np.exp(-1.0 / (release * 1000))
    g = blkm.copy()
    for i in range(1, nb):
        g[i] = min(blkm[i], g[i - 1] * rel + (1 - rel) * blkm[i]) if blkm[i] > g[i - 1] else blkm[i]
    centers = (np.arange(nb) + 0.5) * B
    gs = np.interp(np.arange(len(x)), centers, g)
    gs = np.minimum(gs, np.repeat(blkm, B)[: len(x)])
    return (x * gs[:, None]).astype(np.float32)


def finalize(x, sr, lufs=-14.0, ceiling_db=-1.0, loud_fn=None, tp_fn=None):
    """Gain to the target integrated loudness, then true-peak limiting (iterated to land on target)."""
    for _ in range(4):
        cur = loud_fn(x, sr)
        if not np.isfinite(cur):
            break
        x = x * undb(lufs - cur)
        x = limiter(x, sr, ceiling_db - 0.15)
        if abs(loud_fn(x, sr) - lufs) < 0.3:
            break
    tp = tp_fn(x, sr)
    if tp > ceiling_db:
        x = x * undb(ceiling_db - tp - 0.05)
    return x.astype(np.float32)


# ================================================================ edges
def trim_edges(x, sr, rel_db=-42.0, keep_s=0.6, fade_s=1.5):
    """Cuts noise-only lead-in / run-out (needle drop, tape leader, groove run-out) and fades the edges.
    Threshold: rel_db below the active level. Returns (y, cut_start_s, cut_end_s)."""
    lv = frame_rms_db(x.mean(1), sr, 0.05, win=0.5)
    thr = active_level(x, sr) + rel_db
    on = np.where(lv > thr)[0]
    if not len(on):
        return x, 0.0, 0.0
    a = max(0, int((on[0] * 0.05 - keep_s) * sr))
    b = min(len(x), int(((on[-1] + 1) * 0.05 + keep_s * 2) * sr))
    y = x[a:b].copy()
    fi = min(int(0.02 * sr), len(y) // 4) if a == 0 else min(int(0.3 * sr), len(y) // 4)
    fo = min(int(fade_s * sr), len(y) // 4) if b < len(x) else min(int(0.05 * sr), len(y) // 4)
    if fi:
        y[:fi] *= np.linspace(0, 1, fi)[:, None] ** 2
    if fo:
        y[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
    return y.astype(np.float32), a / sr, (len(x) - b) / sr


# ================================================================ tape azimuth
def detect_azimuth(x, sr=SR, seconds=90):
    """Inter-channel delay (samples, fractional) of a stereo tape with a misaligned head, measured on the
    2-8 kHz band where azimuth error hurts. Positive = right channel late. Returns (lag, correlation)."""
    seg = x[: int(seconds * sr)]
    if seg.shape[1] < 2:
        return 0.0, 1.0
    b = ss.sosfiltfilt(ss.butter(4, [2000, 8000], "bandpass", fs=sr, output="sos"), seg, axis=0)
    L, R = b[:, 0], b[:, 1]
    maxlag = int(0.0015 * sr)
    c = ss.correlate(R, L, mode="full", method="fft")
    mid = len(L) - 1
    w = c[mid - maxlag: mid + maxlag + 1]
    j = int(np.argmax(w))
    if 0 < j < len(w) - 1:
        d = 0.5 * (w[j - 1] - w[j + 1]) / (w[j - 1] - 2 * w[j] + w[j + 1] + 1e-12)
    else:
        d = 0.0
    corr = float(w[j] / (np.sqrt(np.dot(L, L) * np.dot(R, R)) + 1e-12))
    return float(j - maxlag + d), corr


def fix_azimuth(x, lag):
    """Shift the right channel by -lag samples (fractional, via FFT phase) so both channels line up."""
    if abs(lag) < 0.25:
        return x
    n = len(x)
    N = 1 << int(np.ceil(np.log2(n + 64)))
    f = np.fft.rfftfreq(N)
    R = np.fft.rfft(x[:, 1], N) * np.exp(2j * np.pi * f * lag)
    y = x.copy()
    y[:, 1] = np.fft.irfft(R, N)[:n]
    return y.astype(np.float32)


# ================================================================ low end
def bass_restore(x, sr, amount=1.0, lo=70.0, hi=180.0):
    """Sub-harmonic synthesis (like the dbx 120 'Boom Box'): the 70-180 Hz band of the bass / kick is turned into a
    square wave one octave lower (flip-flop on positive zero crossings, gated so noise does not toggle it), shaped by
    the band's envelope and low-passed. It rebuilds the 35-90 Hz fundamentals that worn records, cheap cassettes and
    small home recorders lost completely. Returns (y, level_db of the added sub)."""
    if amount <= 0:
        return x, None
    m = x.mean(1).astype(np.float64)
    band = ss.sosfiltfilt(ss.butter(4, [lo, hi], "bandpass", fs=sr, output="sos"), m)
    env = np.sqrt(np.maximum(uniform_filter1d(band ** 2, int(0.012 * sr)), 0)) * np.sqrt(2)
    act = np.percentile(env, 95)
    gate = env > act * 0.08                                  # -22 dB below the loud part
    sgn = np.where(gate, np.sign(band), 0)
    up = np.where((sgn[1:] > 0) & (sgn[:-1] <= 0))[0] + 1    # positive-going crossings
    tog = np.zeros(len(m), np.int8)
    tog[up] = 1
    sq = (np.cumsum(tog) % 2) * 2.0 - 1.0
    sub = sq * env * gate
    sub = ss.sosfiltfilt(ss.butter(4, 95, "lp", fs=sr, output="sos"), sub)
    sub = ss.sosfiltfilt(ss.butter(2, 28, "hp", fs=sr, output="sos"), sub)
    # level: about the band's own level (a full-range bass has its fundamental ~ as loud as the 1st harmonic)
    g = np.sqrt(np.mean(band[gate] ** 2) + 1e-20) / (np.sqrt(np.mean(sub[gate] ** 2)) + 1e-12) if gate.any() else 0.0
    sub = (sub * g * 0.8 * amount).astype(np.float32)
    y = x + sub[:, None]
    return y.astype(np.float32), round(float(20 * np.log10(g * 0.8 * amount + 1e-12)), 1)


def find_edges(x, sr, gap_s=0.7, quiet_rel=-45.0, tail_rel=-10.0, min_tail=3.0):
    """Where the music starts and ends (in samples). Besides plain silence it recognises a run-out / lead-in:
    a silent gap (>= gap_s, quiet_rel below the active level) with only low-level noise (tail_rel below the
    active level, >= min_tail s) between the gap and the file edge. -> (start, end)."""
    step = 0.05
    lv = frame_rms_db(x.mean(1), sr, step, win=0.2)
    act = active_level(x, sr)
    n = len(lv)
    silent = lv < act + quiet_rel
    lab, cnt = label(silent)
    a, b = 0, n
    on = np.where(lv > act - 42)[0]
    if len(on):
        a, b = on[0], on[-1] + 1
    gl = int(gap_s / step)
    for i in range(1, cnt + 1):
        idx = np.where(lab == i)[0]
        if len(idx) < gl:
            continue
        s, e = idx[0], idx[-1] + 1
        if (n - e) * step >= min_tail and e < n and np.max(lv[e:]) < act + tail_rel and s * step > 10:
            b = min(b, s)
        if s * step >= min_tail and s > 0 and np.max(lv[:s]) < act + tail_rel and (n - e) * step > 10:
            a = max(a, e)
    return int(a * step * sr), int(min(len(x), b * step * sr))


def apply_edges(x, sr, a, b, keep_s=0.4, fade_in=0.15, fade_out=1.2):
    """Cut [a, b) with a little air around it and fades. Returns (y, cut_start_s, cut_end_s)."""
    a = max(0, a - int(keep_s * sr))
    b = min(len(x), b + int(keep_s * 2 * sr))
    y = x[a:b].copy()
    fi = min(int((fade_in if a > 0 else 0.01) * sr), len(y) // 4)
    fo = min(int((fade_out if b < len(x) else 0.05) * sr), len(y) // 4)
    if fi:
        y[:fi] *= (np.linspace(0, 1, fi) ** 2)[:, None]
    if fo:
        y[-fo:] *= (np.linspace(1, 0, fo) ** 2)[:, None]
    return y.astype(np.float32), a / sr, (len(x) - b) / sr


def octave_down(m, sr, lo=60.0, hi=200.0, gate_rel=0.08):
    """Mono signal -> its sub-octave (flip-flop divider on the lo..hi band, envelope-shaped, 28-100 Hz),
    level-matched to the band. Works best on a monophonic part (a bass stem)."""
    band = ss.sosfiltfilt(ss.butter(4, [lo, hi], "bandpass", fs=sr, output="sos"), m.astype(np.float64))
    env = np.sqrt(np.maximum(uniform_filter1d(band ** 2, int(0.012 * sr)), 0)) * np.sqrt(2)
    gate = env > np.percentile(env, 95) * gate_rel
    sgn = np.where(gate, np.sign(band), 0)
    up = np.where((sgn[1:] > 0) & (sgn[:-1] <= 0))[0] + 1
    tog = np.zeros(len(m), np.int8)
    tog[up] = 1
    if not gate.any() or np.sqrt(np.mean(band ** 2)) < 1e-5:      # no real bass in this stem
        return np.zeros(len(m), np.float32)
    sub = ((np.cumsum(tog) % 2) * 2.0 - 1.0) * env * gate
    sub = ss.sosfiltfilt(ss.butter(4, 100, "lp", fs=sr, output="sos"), sub)
    sub = ss.sosfiltfilt(ss.butter(2, 28, "hp", fs=sr, output="sos"), sub)
    if gate.any():
        sub *= np.sqrt(np.mean(band[gate] ** 2)) / (np.sqrt(np.mean(sub[gate] ** 2)) + 1e-12)
    return sub.astype(np.float32)


def kick_sub(drums, sr, f_start=110.0, f_end=52.0, decay=0.16):
    """Kick reinforcement for recordings that lost their low end: kick hits are found in the drum stem
    (onsets whose energy is mostly below 200 Hz) and a pitched sub thump (sine, 110 -> 52 Hz sweep) is placed
    on each, scaled by the hit's strength. -> (mono sub, number of hits)."""
    m = drums.mean(1) if drums.ndim == 2 else drums
    lowb = ss.sosfiltfilt(ss.butter(4, [45, 200], "bandpass", fs=sr, output="sos"), m)
    hib = ss.sosfiltfilt(ss.butter(2, [1500, 6000], "bandpass", fs=sr, output="sos"), m)
    hop = int(0.005 * sr)
    n = len(m) // hop
    el = np.sqrt((lowb[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9
    eh = np.sqrt((hib[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9
    ld = 20 * np.log10(el)
    flux = np.maximum(0, ld - minimum_filter1d(ld, 7, origin=3))        # rise within the last 30 ms
    peaks, props = ss.find_peaks(flux, height=6.0, distance=int(0.11 / 0.005))
    out = np.zeros(len(m), np.float32)
    if not len(peaks):
        return out, 0
    # strength = low-band peak level within 40 ms after the onset; keep the kick-like ones (low >> high)
    strength = np.array([el[p:p + 8].max() for p in peaks])
    lowness = np.array([20 * np.log10(el[p:p + 8].max() / eh[p:p + 8].max()) for p in peaks])
    keep = (strength > np.percentile(el, 75)) & (lowness > np.median(lowness) - 3)
    peaks, strength = peaks[keep], strength[keep]
    if not len(peaks):
        return out, 0
    L = int(0.45 * sr)
    t = np.arange(L) / sr
    f = f_end + (f_start - f_end) * np.exp(-t / 0.025)
    ph = 2 * np.pi * np.cumsum(f) / sr
    att = np.minimum(1, t / 0.002)
    shape = (np.sin(ph) * np.exp(-t / decay) * att).astype(np.float32)
    ref = np.median(strength)
    for p, s in zip(peaks, strength):
        a = p * hop
        b = min(len(out), a + L)
        out[a:b] += shape[: b - a] * float(min(2.0, s / ref)) * float(ref) * 2.2
    return out, int(len(peaks))



# ================================================================ hiss / rumble guard
def noise_floor(x, sr, n=4096, smooth_s=0.12, pct=3.0, bias_db=2.0):
    """Stationary noise floor per bin (power, mono mix) by minimum statistics (R. Martin, 2001): the power spectrum is
    smoothed over ~0.1 s and a low percentile over time is taken, plus a small bias. Dense music hides the hiss
    from a plain percentile; minimum statistics still finds it."""
    m = x.mean(1) if x.ndim == 2 else x
    hop = n // 4
    _, _, S = ss.stft(m, sr, nperseg=n, noverlap=n - hop)
    P = uniform_filter1d(np.abs(S) ** 2, max(1, int(smooth_s * sr / hop)), axis=1)
    return np.percentile(P, pct, axis=1) * undb(bias_db) ** 2


def median_power(x, sr, n=4096):
    m = x.mean(1) if x.ndim == 2 else x
    _, _, S = ss.stft(m, sr, nperseg=n, noverlap=n - n // 4)
    return np.median(np.abs(S) ** 2, axis=1)


def hiss_guard(x, sr, n0, extra_db=6.0, max_cut_db=24.0):
    """Keeps the enhancement from lifting the noise: n0 is the noise floor (noise_floor()) measured BEFORE the
    EQ / exciter / bass synthesis. Wherever the current noise floor n1 is above n0, a smoothed Wiener gate pulls the
    noise back down to n0 - extra_db, while bins with music well above the floor keep their boost.
    Returns (y, mean reduction of the floor in dB above 2 kHz)."""
    n = 4096
    hop = n // 4
    n1 = noise_floor(x, sr, n)
    f = np.fft.rfftfreq(n, 1 / sr)
    lift = np.clip(n1 / (n0 + 1e-20) * undb(extra_db) ** 2, 1.0, undb(max_cut_db) ** 2)   # power ratio to remove
    lift = uniform_filter1d(lift, 9)
    gmin = 1.0 / np.sqrt(lift)                                                           # deepest allowed cut
    stats = {}

    def gains(f_, t, P):
        Ps = uniform_filter1d(uniform_filter1d(P, 3, axis=0), 5, axis=1)
        N = n1[:, None]
        snr = np.maximum(Ps / (1.5 * N + 1e-20) - 1.0, 0.0)
        g = snr / (snr + 1.0)                                  # Wiener on the post-enhancement floor
        g = np.maximum(np.sqrt(g), gmin[:, None])
        g = uniform_filter1d(uniform_filter1d(g, 3, axis=0), 7, axis=1)
        stats["hf"] = float(-20 * np.log10(np.median(g[f_ > 2000]) + 1e-9)) if (f_ > 2000).any() else 0.0
        return g

    y = spec_process(x, sr, n, hop, gains)
    return y.astype(np.float32), round(stats.get("hf", 0.0), 1)



# ================================================================ vinyl: vertical (side) noise, mono records
def side_denoise(x, sr, reduction_db=9.0):
    """Surface noise of a record is mostly vertical groove motion = the L-R (side) signal, while the music sits mostly
    in the lateral (mid) signal. The side channel gets its own spectral noise reduction (minimum-statistics floor,
    Wiener gain limited to reduction_db: ~9 dB is clean, beyond ~12 dB the stereo image starts to sound odd)."""
    if reduction_db <= 0 or x.shape[1] < 2:
        return x
    mid = (x[:, 0] + x[:, 1]) / 2
    side = ((x[:, 0] - x[:, 1]) / 2)[:, None].astype(np.float32)
    n0 = noise_floor(side, sr)
    floor = undb(-reduction_db)

    def gains(f, t, P):
        Ps = uniform_filter1d(uniform_filter1d(P, 3, axis=0), 5, axis=1)
        snr = np.maximum(Ps / (1.5 * n0[:, None] + 1e-20) - 1.0, 0.0)
        g = np.maximum(np.sqrt(snr / (snr + 1.0)), floor)
        return uniform_filter1d(uniform_filter1d(g, 3, axis=0), 7, axis=1)

    s2 = spec_process(side, sr, 4096, 1024, gains)[:, 0]
    return np.stack([mid + s2, mid - s2], 1).astype(np.float32)


def mono_sum(x):
    """A mono recording transferred in stereo: both channels carry the same music but partly independent noise and
    one-sided clicks, so their average is ~3 dB quieter in noise (~6 dB for clicks on one groove wall)."""
    return np.repeat(x.mean(1, keepdims=True), 2, 1).astype(np.float32)
