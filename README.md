# Recover Studio

**Restore old, damaged recordings in your browser and download the result as WAV.**
Drop in a digitized vinyl record, cassette, reel-to-reel tape, shellac 78, an old MP3 or a home demo. The studio
finds what is wrong with it, repairs it, rebuilds what the medium lost and masters it, like LANDR does for new mixes.
The goal is restoration: the result should sound like the original recording, not like a new song.

Everything runs locally (Windows, Linux, macOS or Google Colab) and the recording is never uploaded anywhere.
The software is portable Python: one folder, nothing installed system-wide. It detects the hardware automatically
(NVIDIA CUDA, AMD ROCm, Apple MPS, Intel XPU or plain CPU) and works on all of them.

> Russian guide: [README.txt](README.txt)

## How it works (Analyze → Repair → Enhance → Master)

The chain follows how restoration is done in professional studios (CEDAR / iZotope RX practice): most destructive
corrections first (clicks before crackle before hiss), noise reduction with a limit ("less is more"), vertical vinyl
noise handled in the side channel, and never boosting a band that is mostly noise.

The workflow follows LANDR: upload, automatic analysis, a short preview, choose a style and intensity, then render
and download.

1. **Diagnosis** starts as soon as a file is uploaded: mains hum (50/60 Hz and harmonics), clicks and crackle per
   minute, signal-to-noise ratio, upper/lower bandwidth, tonal deviation from a reference curve, wow & flutter (rate,
   depth, periodicity), clipping, tape azimuth, mono/stereo, and silent or noise-only lead-in/run-out. It also
   guesses the medium (vinyl, shellac, cassette, reel, lossy file or demo).
2. **Repair**, each stage set to *Auto / On / Off* (Auto turns on only the stages the diagnosis calls for):

   | Stage | Method |
   |---|---|
   | Mono sources | a mono disc or tape transferred in stereo is summed to mono first: the music is identical in both channels, the noise is not (about −3 dB of noise, −6 dB for one-wall clicks) |
   | Rumble, DC | zero-phase high-pass (25 Hz for records, 20 Hz otherwise) |
   | Hum | detected fundamental (exact Hz) + harmonics removed with narrow zero-phase notches |
   | Clicks / crackle | AR-residual detection with an isolation test (drum hits are not clicks) + least-squares AR interpolation (Godsill & Rayner) |
   | Clipping | clipped runs redrawn by LSAR, never below the clip level |
   | Wow & flutter | speed deviation tracked from the whole-mix log-frequency spectrum, removed by time-varying resampling |
   | Tape azimuth | inter-channel delay measured at 2–8 kHz, corrected with a fractional shift |
   | Record surface noise | for stereo vinyl / shellac the side (L−R) channel, which carries the vertical groove noise, gets its own noise reduction, capped at 9 dB |
   | Noise | **Mel-Band RoFormer Denoise** (aufr33), a neural network trained to remove noise from music (normal and aggressive models); spectral Wiener denoiser on CPU. On by default for anything with less than 40 dB SNR, because later EQ would otherwise expose the hiss |
3. **Enhance**:

   | Stage | Method |
   |---|---|
   | Low end | **BS-RoFormer SW** splits the track into 6 stems: the bass stem gets its sub-octave back, kick hits in the drum stem get a pitched sub thump. On CPU, a sub-harmonic synthesizer is used instead |
   | Highs | harmonic exciter regenerates the band above the cut-off measured after denoising; it is gated, so it only works where there is music in the band below, not hiss |
   | Codec artifacts | **Apollo** (Tsinghua / Tencent AI Lab), a generative model for MP3 and other lossy sources |
   | Tonal balance | linear-phase EQ matches the long-term spectrum to a reference curve of finished masters, or to **your reference track** (like LANDR Reference). A band is boosted only where its median content is at least 8 dB above its noise floor, so a band that is mostly hiss or rumble is never raised |
   | Noise guard | the noise floor is measured (minimum statistics, R. Martin) after the repair and again after all enhancement; a smoothed Wiener gate pulls any raised hiss/rumble back to 6 dB below the post-repair floor, while music above the floor keeps its boost |
   | Stem balance (optional) | level of vocals / drums / bass / guitar / keys / other, vocal de-reverb (anvuew), downloadable stems |
   | Stereo | mid/side width with mono bass |
4. **Master**: character (*Warm / Balanced / Open*), intensity (*Low / Medium / High*: −16 / −14 / −11 LUFS), glue
   compression, tape saturation, 4× oversampled true-peak limiter at −1 dBTP, then trimming of the noise-only
   run-out with fades.

Results: the **restored master**, a **repair-only** version (defects removed, tone and level untouched, for your own
mix or master), stems if enabled, and a text report. WAV export at 44.1 / 48 / 96 kHz, 16 bit (dithered),
24 bit or 32-bit float.

### The web studio

- **Diagnosis**: metric tiles, a list of findings by severity, the tonal curve against the target, a spectrogram.
- **A/B compare**: sample-accurate switching between *A original / B restored / C repair only* while playing (keys
  1/2/3, Space). *Match loudness* is on by default, so "louder" doesn't pass for "better". You also get a before/after
  table, before/after/target curves, before/after spectrograms and the list of what was done.
- **30-second preview**: the same chain on an excerpt (the loudest part, or a start time you choose), done in seconds.
- **Download WAV**: the restored master, the repair-only version, or everything as a ZIP.
- **Projects**: every upload is kept in `projects/<date>_<name>/`.

## Installation

| System | Command |
|---|---|
| Windows 10/11 | `install.bat` installs everything into this folder (about 5 GB with CUDA; internet is needed once) |
| Linux / macOS | `bash install.sh` |
| Google Colab | open `colab/RecoverStudio_Colab.ipynb` and run steps 1–4 |

The installer picks the PyTorch build: NVIDIA → CUDA 12.6, AMD (Linux) → ROCm, Apple Silicon → MPS, Intel Arc → XPU,
anything else → CPU. `install.bat cpu` forces the CPU build. CUDA 12.6 needs NVIDIA driver 560 or newer.
The installer also creates a *Recover Studio* shortcut (set `RCS_NO_SHORTCUT=1` to skip it).

It downloads Denoise (870 MB), Apollo (66 MB) and BS-RoFormer SW (700 MB) into `models/`. The aggressive denoiser and
the vocal de-reverb download on first use, or right away with `set RCS_MODELS_PRELOAD=denoise_aggr dereverb`.

## Running

| Command | What it does |
|---|---|
| `RecoverStudio.bat` (Linux/macOS: `bash run.sh`) | Starts the web studio at http://127.0.0.1:8793/ |
| `recover.bat "side A.wav"` | Restores a file from the command line with automatic settings. You can also drag a file onto it |
| `check-gpu.bat` | Shows which device will be used, and why |

`recover.bat` / `app/recover.py` options: `--source auto|vinyl|shellac|cassette|reel|lossy|demo`,
`--style warm|balanced|open`, `--intensity low|medium|high`, `--lufs -14`, `--reference ref.wav`, `--preview`,
`--stems --vocals 2 --dereverb`, `--apollo on|off|auto`, `--engine auto|ai|dsp`, `--device auto|cuda|cpu`,
`--sr 44100|48000|96000`, `--bits 16|24|32`, `--out folder`.

### GPU, CPU and speed

*Neural networks: Auto* uses the networks on a GPU and fast DSP on a CPU, so every machine gives a result in
reasonable time. *Always* uses the networks on a CPU too (best quality, slow). *DSP only* is the fastest.

Measured on an RTX 4060 Ti 16 GB (CUDA 12.6) with a Ryzen 7 5800X:

| Job | Time |
|---|---|
| 2:29 lossy home demo, GPU, full chain (denoise, stems low end, Apollo, exciter, noise guard, master) | 101 s |
| 3:33 lossy home demo, GPU, full chain | 138 s |
| 5:02 vinyl transfer, GPU, full chain (side-channel denoise, AI denoise, stems low end, noise guard, master) | 198 s |
| 5:02 vinyl transfer, CPU / DSP mode (*Neural networks: Auto* without a GPU) | 143 s |
| 30-second preview, GPU (mostly model loading) | ≈ 30 s |
| 30-second preview, CPU / DSP mode | 32 s |

Peak VRAM per network (one in memory at a time): Denoise 4.5 GB, BS-RoFormer stems 4.3 GB, Apollo 9.6 GB (reserved,
batch of 4 windows). Everything fits in 16 GB; cards under 14 GB get smaller batches automatically.

What keeps it fast on 16 GB: fp16/bf16 autocast with an automatic fp32 retry for windows that overflow; several
windows per GPU batch (smaller batches and then the CPU if memory runs out); only one network in memory at a time,
unloaded after each job; analysis cached per project; all DSP vectorised with NumPy/SciPy.

## Google Colab

`colab/RecoverStudio_Colab.ipynb` runs the same web studio on a free Colab GPU (T4):

1. **Check GPU** (Runtime → Change runtime type → T4 GPU).
2. **Code & Google Drive**: `git` with this repository's URL, or upload `recover-studio-app.zip`
   (built by `python tools/make_dist.py`). With Drive storage on, projects and models persist between sessions.
3. **Install**: about 3–5 minutes, once per session.
4. **Run studio**: `TUNNEL = colab` (Colab proxy), `ngrok` (public link: paste a token from
   https://dashboard.ngrok.com or save it in Colab Secrets as `NGROK_AUTHTOKEN`; optional static domain and
   password) or `both`.
5. Optional: restore a file without the UI and download the WAV.

The notebook is generated by `tools/make_colab.py`. Edit that script, not the `.ipynb`. After publishing the
repository, put its URL into `REPO_URL` in `tools/make_colab.py` and run the script again.

## Project layout

| Path | Contents |
|---|---|
| `app/server.py` | Web server and JSON API (standard library only) |
| `app/recover.py` | Command line |
| `app/studio/pipeline.py` | The chain, jobs, projects, report |
| `app/studio/analyze.py` | Damage diagnosis and auto settings |
| `app/studio/dsp.py` | Restoration and mastering DSP |
| `app/studio/roformer.py`, `apollo.py` | Neural networks (pure PyTorch inference) |
| `app/studio/models.py`, `device.py`, `audio_io.py` | Model registry/downloader, hardware detection, audio I/O |
| `app/web/` | Single-page UI (Tailwind bundle in `vendor/`) |
| `tools/` | `make_colab.py` (notebook), `make_dist.py` (code ZIP) |
| `colab/` | Google Colab notebook |

These folders aren't stored in the repository. The installers create them, or they're generated:
`python/` or `.venv/`, `models/`, `projects/` and `dist/`.

## Models

| Model | Used for | Author / license |
|---|---|---|
| Mel-Band RoFormer Denoise (+ Aggr) | noise removal | aufr33 (UVR community) |
| BS-RoFormer SW, 6 stems | stems, low-end rebuild | jarredou / undef13 |
| Apollo | lossy-codec restoration | Kai Li, Yi Luo, CC BY-SA 4.0 |
| Mel-Band RoFormer De-Reverb | vocal room removal (optional) | anvuew, GPL-3.0 |

See `THIRD_PARTY_NOTICES.txt` for details. The weights belong to their authors. Check their terms before you use the
results commercially.

## Limitations

- Restoration can't bring back what is fully gone. Missing highs and sub-bass are *re-synthesised* (exciter,
  sub-octave, kick reinforcement), so they are plausible, not the original signal. For the closest result, give a
  **reference** (a good release of the same song, or a similar record).
- Wow correction assumes the music itself holds pitch. Strong vibrato or pitch bends across the whole mix can be
  read as speed drift, so lower *strength* or switch the stage off.
- The click detector protects drum attacks, but very sharp percussion at high sensitivity can be touched. Compare A/B.
- Only restore recordings you have the rights to.
