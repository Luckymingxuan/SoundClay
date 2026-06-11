# SoundClay Engine

v1.2 adds an experimental sample-based instrument export:

```text
WAV audio -> Basic Pitch -> MIDI
          -> isolated note samples -> SFZ -> ZIP
```

The MIDI transcription backend uses Spotify's open-source Basic Pitch model. The
instrument label is currently lightweight routing metadata; a dedicated
classifier is not required for SFZ generation.

The instrument ZIP contains the MIDI file, an SFZ mapping, and the extracted WAV
samples. Input works best when notes are held for at least 0.35 seconds and do
not overlap heavily.

```bash
.venv/bin/python engine/soundclay_engine.py --input piano.wav --output-dir out
```
