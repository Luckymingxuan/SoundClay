# SoundClay Engine

v1.1 focuses on the smallest useful loop:

```text
WAV audio -> instrument routing metadata -> Basic Pitch -> MIDI
```

The MIDI transcription backend uses Spotify's open-source Basic Pitch model. The
instrument label is currently lightweight routing metadata; a dedicated
open-source classifier should replace it next.

```bash
.venv/bin/python engine/soundclay_engine.py --input piano.wav --output-dir out
```
