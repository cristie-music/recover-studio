"""Damage analysis ("what is wrong with this recording") and automatic settings, like LANDR's analysis step.

analyze(x) -> report dict (all numbers JSON-friendly) with a list of findings for the UI.
auto_options(report, source, options) -> options with every 'auto' switch resolved.
measure(x) -> compact before/after metrics.
"""
import numpy as np

from . import audio_io as A
from . import dsp

SOURCES = {
    "auto": dict(title="Авто (определить)", about="Студия сама определит дефекты и включит нужные модули."),
    "vinyl": dict(title="Винил (LP / сингл)", about="Щелчки, треск, шум поверхности, гул, рокот, детонация пластинки."),
    "shellac": dict(title="Шеллак 78 об/мин", about="Сильный треск и шум, узкая полоса, часто моно."),
    "cassette": dict(title="Кассета", about="Шипение ленты, завал верхов, детонация (wow/flutter), азимут головки."),
    "reel": dict(title="Катушка / бобина", about="Шипение, гул, азимут, лёгкая детонация."),
    "lossy": dict(title="MP3 / сжатый файл", about="Артефакты кодека, срезанные верха (Apollo)."),
    "demo": dict(title="Домашнее демо / репетиция", about="Шум комнаты, гулкость, глухой звук, неровный баланс."),
}


def analyze(x, sr=A.SR, src_ext="", channels=2, src_sr=44100, log=print):
    """x float32 (n, 2). Returns dict with metrics and human-readable findings."""
    dur = len(x) / sr
    lufs = A.loudness(x, sr)
    tp = A.true_peak_db(x, sr)
    hum = dsp.detect_hum(x, sr)
    clicks = dsp.count_clicks(x, sr, 5.0)
    crackle = dsp.count_clicks(x, sr, 8.0)
    wow = dsp.analyze_wow(x, sr)
    fc = dsp.detect_cutoff(x, sr)
    lowc = dsp.detect_lowcut(x, sr)
    corr = dsp.stereo_corr(x)
    mono = channels == 1 or corr > 0.995
    az_lag, az_corr = dsp.detect_azimuth(x, sr) if not mono else (0.0, 1.0)
    clip = dsp.clip_ratio(x)
    cur, noise = dsp.band_curve(x, sr)
    sel = (dsp.GRID > 60) & (dsp.GRID < 12000)
    snr = float(np.median((cur - noise)[sel]))
    tone_err = dsp.norm_curve(cur) - dsp.TARGET_BALANCED
    sel2 = (dsp.GRID > 40) & (dsp.GRID < 14000)
    low_def = float(np.mean(tone_err[(dsp.GRID > 40) & (dsp.GRID < 150)]))
    high_def = float(np.mean(tone_err[(dsp.GRID > 5000) & (dsp.GRID < 14000)]))
    lv = dsp.frame_rms_db(x.mean(1), sr, 0.05, win=0.5)
    act = dsp.active_level(x, sr)
    quiet = lv < act - 42
    lead = float(np.argmax(~quiet) * 0.05) if (~quiet).any() else 0.0
    tail = float((len(quiet) - 1 - np.max(np.where(~quiet)[0])) * 0.05) if (~quiet).any() else 0.0

    F = []   # findings: (severity 0..3, key, text)

    def add(sev, key, text):
        F.append(dict(severity=sev, key=key, text=text))

    if hum.get("f0"):
        add(2 if hum["score"] > 50 else 1, "hum",
            f"Гул сети {hum['f0']:.0f} Гц ({hum['freq']:.2f} Гц) с гармониками: {', '.join(str(n) for n, _ in hum['harmonics'])}")
    if clicks > 150:
        add(3 if clicks > 1500 else 2 if clicks > 500 else 1, "clicks", f"Щелчки: ~{clicks:.0f} в минуту")
    if crackle > 1500 and crackle > 2.5 * max(clicks, 1):
        add(2, "crackle", f"Мелкий треск: ~{crackle:.0f} событий в минуту")
    if wow["periodicity"] > 0.13 and wow["wow_cents"] > 0.9:
        add(2 if wow["wow_cents"] > 3 else 1, "wow",
            f"Детонация (плавание скорости): ±{wow['wow_cents']:.1f} цента, частота {wow['rate_hz']} Гц")
    elif wow["flutter_cents"] > 1.2:
        add(1, "wow", f"Флаттер (дрожание скорости): ±{wow['flutter_cents']:.1f} цента")
    if snr < 30:
        add(3 if snr < 15 else 2 if snr < 22 else 1, "noise", f"Шум: сигнал/шум ≈ {snr:.0f} дБ")
    if fc < 15500:
        add(2 if fc < 11000 else 1, "bandwidth", f"Срезанные верха: полоса до ~{fc / 1000:.1f} кГц")
    if high_def < -6:
        add(2 if high_def < -12 else 1, "dull", f"Глухой звук: верха на {-high_def:.0f} дБ ниже нормы")
    if low_def < -6:
        add(2 if low_def < -12 else 1, "thin", f"Мало низа: {-low_def:.0f} дБ ниже нормы")
    if clip > 2e-4:
        add(2 if clip > 2e-3 else 1, "clip", f"Клиппинг: {clip * 100:.2f}% семплов в срезанных пиках")
    if mono:
        add(0, "mono", "Запись моно (или почти моно)")
    if abs(az_lag) > 0.6 and az_corr > 0.3:
        add(1, "azimuth", f"Азимут: правый канал сдвинут на {az_lag:+.1f} семпла ({az_lag / sr * 1e6:+.0f} мкс)")
    if src_ext in (".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma"):
        add(1, "lossy", f"Сжатый формат {src_ext[1:].upper()}: возможны артефакты кодека")
    if lead > 1.5 or tail > 1.5:
        add(0, "edges", f"Тишина/шум без музыки: в начале {lead:.1f} с, в конце {tail:.1f} с")
    if not F:
        add(0, "ok", "Серьёзных дефектов не найдено: студия выполнит бережную реставрацию и мастеринг")

    # guess the carrier
    guess = "demo"
    if (clicks > 400 or crackle > 1500) and (hum.get("f0") or wow["periodicity"] > 0.13 or fc < 14000):
        guess = "shellac" if (fc < 8000 and mono) else "vinyl"
    elif src_ext in (".mp3", ".ogg", ".m4a", ".aac", ".opus", ".wma") and clicks < 300 and snr > 30:
        guess = "lossy"
    elif snr < 30 and (wow["wow_cents"] > 1.2 or fc < 14000) and clicks < 400:
        guess = "cassette"
    elif hum.get("f0") and clicks < 400:
        guess = "reel"

    return dict(duration=round(dur, 2), lufs=round(lufs, 1) if np.isfinite(lufs) else None, true_peak=round(tp, 1),
                hum=hum, clicks_per_min=round(clicks), crackle_per_min=round(crackle), wow=wow,
                cutoff_hz=round(fc), lowcut_hz=round(lowc), stereo_corr=round(corr, 3), mono=bool(mono),
                azimuth_lag=round(az_lag, 2), azimuth_corr=round(az_corr, 3), clip_ratio=round(clip, 6),
                snr_db=round(snr, 1), low_deficit_db=round(low_def, 1), high_deficit_db=round(high_def, 1),
                lead_quiet_s=round(lead, 1), tail_quiet_s=round(tail, 1), source_guess=guess,
                source_sr=src_sr, channels=channels, active_db=round(act, 1),
                curve=[round(float(v), 1) for v in dsp.norm_curve(cur)],
                noise_curve=[round(float(v), 1) for v in (noise - (cur - dsp.norm_curve(cur)))],
                findings=sorted(F, key=lambda d: -d["severity"]))


def measure(x, sr=A.SR):
    """Compact metrics for the before/after table."""
    cur, noise = dsp.band_curve(x, sr)
    sel = (dsp.GRID > 60) & (dsp.GRID < 12000)
    lufs = A.loudness(x, sr)
    hum = dsp.detect_hum(x, sr)
    return dict(lufs=round(lufs, 1) if np.isfinite(lufs) else None, true_peak=round(A.true_peak_db(x, sr), 1),
                snr_db=round(float(np.median((cur - noise)[sel])), 1), clicks_per_min=round(dsp.count_clicks(x, sr)),
                hum_score=hum.get("score", 0) if hum.get("f0") else 0, cutoff_hz=round(dsp.detect_cutoff(x, sr)),
                stereo_corr=round(dsp.stereo_corr(x), 3),
                tone_error_db=round(float(np.mean(np.abs((dsp.norm_curve(cur) - dsp.TARGET_BALANCED)[
                    (dsp.GRID > 40) & (dsp.GRID < 14000)]))), 1),
                curve=[round(float(v), 1) for v in dsp.norm_curve(cur)])


def _on(v):
    return v is True or v == "on"


def auto_options(rep, o, log=print):
    """Resolve every 'auto' in the options from the analysis report and the chosen source type.
    Returns a new dict with booleans + a list of reasons for the log / UI."""
    o = {k: (dict(v) if isinstance(v, dict) else v) for k, v in o.items()}
    src = o.get("source", "auto")
    if src == "auto":
        src = rep["source_guess"]
    o["source_resolved"] = src
    keys = {f["key"] for f in rep["findings"]}
    why = []

    def decide(mod, cond, reason):
        m = o[mod]
        if m.get("on") == "auto":
            m["on"] = bool(cond)
            if cond:
                why.append(f"{mod}: {reason}")
        else:
            m["on"] = _on(m.get("on"))

    decide("hum", "hum" in keys, "найден гул")
    decide("clicks", "clicks" in keys or src in ("vinyl", "shellac"), "щелчки / виниловый носитель")
    decide("crackle", "crackle" in keys or src == "shellac", "мелкий треск")
    decide("declip", "clip" in keys, "клиппинг")
    decide("wow", "wow" in keys, "детонация")
    decide("azimuth", "azimuth" in keys and src in ("cassette", "reel", "auto", "demo"), "азимут ленты")
    decide("denoise", "noise" in keys or rep["snr_db"] < 40 or src in ("vinyl", "shellac", "cassette", "reel"),
           "шум / шипение")
    decide("apollo", "lossy" in keys or src == "lossy", "сжатый исходник")
    if o["bandwidth"].get("on") == "auto":
        o["bandwidth"]["on"] = "auto"      # decided after denoising: the real music bandwidth shows only without hiss
    else:
        o["bandwidth"]["on"] = _on(o["bandwidth"].get("on"))
    decide("stems", False, "")
    decide("tone", True, "тональный баланс")
    decide("bass", rep["low_deficit_db"] < -10, f"низ на {-rep['low_deficit_db']:.0f} дБ ниже нормы")
    if o["trim"].get("on") == "auto":
        o["trim"]["on"] = "auto"          # decided after denoising (find_edges)
    else:
        o["trim"]["on"] = _on(o["trim"].get("on"))
    o["master"]["on"] = _on(o["master"].get("on", True)) or o["master"].get("on") == "auto"
    # carrier-specific defaults
    if src == "shellac":
        o["crackle"].setdefault("sensitivity", 6)
        o["denoise"]["model"] = o["denoise"].get("model") if o["denoise"].get("model") not in (None, "auto") else "denoise_aggr"
    if o["denoise"].get("model") in (None, "auto"):
        o["denoise"]["model"] = "denoise_aggr" if rep["snr_db"] < 15 else "denoise"
    if o.get("rumble_hz") in (None, "auto"):
        o["rumble_hz"] = 25 if src in ("vinyl", "shellac") else 20
    return o, why
