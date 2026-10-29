# Creating narrated videos

Toolchain validated on this machine (Linux, KDE/X11, Intel + RTX 4060).

## Tools

| Job | Tool | Notes |
|-----|------|-------|
| Edit / PiP / mux / transcode | **ffmpeg** (4.4.2) | Everything non-interactive: trim, overlay, concat, resize, audio mux |
| Synthetic narration (TTS) | **Kokoro** (open-source, neural) | Installed in venv, ~50 voices, runs on CPU (few sec/clip) |
| Play audio to check | **ffplay** | `ffplay -nodisp -autoexit file.mp3` |

Kokoro venv:
`/tmp/.../scratchpad/kokoro-venv` (rebuild with `pip install kokoro soundfile`; `espeak-ng` installed for out-of-dictionary words).

## Narration: script → voice

```python
from kokoro import KPipeline
import soundfile as sf, numpy as np
p = KPipeline(lang_code='a')                    # 'a'=US English, 'b'=UK
audio = np.concatenate([a for _,_,a in p(TEXT, voice='af_heart')])
sf.write('vo.wav', audio, 24000)
```

Good voices: `af_heart` (US female, warm), `am_michael` (US male, narrator), `bf_emma` (UK female).

## Attach narration to video

```bash
# Replace audio with narration
ffmpeg -i video.mp4 -i vo.wav -c:v copy -c:a aac -map 0:v -map 1:a -shortest out.mp4

# Mix narration over ducked original audio
ffmpeg -i video.mp4 -i vo.wav -filter_complex \
  "[0:a]volume=0.3[bg];[bg][1:a]amix=inputs=2:duration=first[a]" \
  -map 0:v -map "[a]" -c:v copy out.mp4
```

## Picture-in-picture

```bash
# Static inset, top-right, 480px wide
ffmpeg -i main.mp4 -i cam.mp4 -filter_complex \
  "[1:v]scale=480:-1[pip];[0:v][pip]overlay=W-w-20:20" out.mp4

# Moving inset (x/y accept time var t); animated size -> use zoompan
```

## Recording gotcha (this machine)

Screen recordings of **Chromium/Brave** windows come out **black** on X11: the browser paints
through a GPU surface the X11 grab can't read (only the cursor survives). Fix before recording:

- Brave: `brave://settings/system` -> turn OFF "Use graphics acceleration when available", relaunch.
- Or launch with `--disable-gpu-compositing` (escalate to `--disable-gpu` if still black).

Same class of bug hits any GPU-accelerated / fullscreen app; record windowed with compositing on,
or use OBS (XSHM/Window Capture) instead of a plain X11 grab.
