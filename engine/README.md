# SoundClay Engine

v1.1 focuses on the smallest useful loop:

```text
WAV audio -> instrument classification -> backend selection -> MIDI
```

The current backend is dependency-free and heuristic. It is meant to keep the app
usable while leaving clear seams for model-backed transcribers later.

```bash
python3 engine/soundclay_engine.py --input piano.wav --output-dir out
```
