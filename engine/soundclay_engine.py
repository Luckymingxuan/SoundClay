#!/usr/bin/env python3
"""SoundClay v1.1 audio-to-MIDI engine.

The production path uses Spotify's open-source Basic Pitch model for MIDI
transcription. Lightweight local analysis is still used for instrument routing
metadata until we add a dedicated open-source instrument classifier.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional


SUPPORTED_EXTENSIONS = {".wav"}

MODEL_BY_INSTRUMENT = {
    "Piano": "basic-pitch-icassp-2022-piano-route",
    "Guitar": "basic-pitch-icassp-2022-guitar-route",
    "Bass": "basic-pitch-icassp-2022-bass-route",
    "Vocal": "basic-pitch-icassp-2022-vocal-route",
    "Synth": "basic-pitch-icassp-2022-synth-route",
    "Strings": "basic-pitch-icassp-2022-strings-route",
}

@dataclass
class AudioData:
    sample_rate: int
    samples: list[float]


def read_wav(path: Path) -> AudioData:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("v1.1 currently supports WAV input. Please export your audio as .wav.")

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        values = [(byte - 128) / 128.0 for byte in frames]
    elif sample_width == 2:
        values = [value / 32768.0 for (value,) in struct.iter_unpack("<h", frames)]
    elif sample_width == 4:
        values = [value / 2147483648.0 for (value,) in struct.iter_unpack("<i", frames)]
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    if channels > 1:
        mono = []
        for index in range(0, len(values), channels):
            mono.append(sum(values[index : index + channels]) / channels)
        values = mono

    return AudioData(sample_rate=sample_rate, samples=values)


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0] >= 0
    for sample in samples[1:]:
        current = sample >= 0
        if current != previous:
            crossings += 1
        previous = current
    return crossings / (len(samples) - 1)


def estimate_pitch(samples: list[float], sample_rate: int) -> Optional[float]:
    energy = rms(samples)
    if energy < 0.01:
        return None

    min_freq = 55.0
    max_freq = 1046.0
    min_lag = max(1, int(sample_rate / max_freq))
    max_lag = min(int(sample_rate / min_freq), len(samples) - 1)
    if max_lag <= min_lag:
        return None

    best_lag = 0
    best_score = 0.0
    for lag in range(min_lag, max_lag):
        score = 0.0
        limit = len(samples) - lag
        step = max(1, limit // 900)
        for index in range(0, limit, step):
            score += samples[index] * samples[index + lag]
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_lag == 0 or best_score <= 0:
        return None
    return sample_rate / best_lag


def analyze_frames(audio: AudioData) -> list[dict[str, float | None]]:
    frame_size = max(512, int(audio.sample_rate * 0.06))
    hop_size = max(256, int(audio.sample_rate * 0.03))
    frames = []

    for start in range(0, max(0, len(audio.samples) - frame_size), hop_size):
        chunk = audio.samples[start : start + frame_size]
        frames.append(
            {
                "time": start / audio.sample_rate,
                "rms": rms(chunk),
                "zcr": zero_crossing_rate(chunk),
                "pitch": estimate_pitch(chunk, audio.sample_rate),
            }
        )

    return frames


def classify_instrument(path: Path, frames: list[dict[str, float | None]]) -> tuple[str, float, dict[str, float]]:
    filename = path.stem.lower()
    name_hints = {
        "piano": "Piano",
        "keys": "Piano",
        "guitar": "Guitar",
        "bass": "Bass",
        "vocal": "Vocal",
        "voice": "Vocal",
        "synth": "Synth",
        "string": "Strings",
        "violin": "Strings",
        "cello": "Strings",
    }
    for hint, instrument in name_hints.items():
        if hint in filename:
            return instrument, 0.9, {"filename_hint": 1.0}

    active = [frame for frame in frames if (frame["rms"] or 0) > 0.02]
    if not active:
        return "Piano", 0.25, {"active_ratio": 0.0}

    pitches = [frame["pitch"] for frame in active if frame["pitch"]]
    zcr_values = [float(frame["zcr"] or 0) for frame in active]
    rms_values = [float(frame["rms"] or 0) for frame in active]
    median_pitch = median(pitches) if pitches else 0.0
    median_zcr = median(zcr_values)
    dynamic_range = max(rms_values) - min(rms_values)
    active_ratio = len(active) / max(1, len(frames))

    if median_pitch and median_pitch < 130:
        instrument = "Bass"
    elif median_zcr > 0.18:
        instrument = "Synth"
    elif active_ratio > 0.7 and dynamic_range < 0.12:
        instrument = "Strings"
    elif median_pitch and 160 <= median_pitch <= 520 and dynamic_range < 0.18:
        instrument = "Vocal"
    elif dynamic_range > 0.18:
        instrument = "Guitar"
    else:
        instrument = "Piano"

    confidence = 0.42
    if pitches:
        confidence += 0.18
    if dynamic_range > 0.08:
        confidence += 0.12
    if active_ratio > 0.25:
        confidence += 0.08

    return (
        instrument,
        min(0.78, confidence),
        {
            "median_pitch": round(median_pitch, 2),
            "median_zero_crossing_rate": round(median_zcr, 4),
            "dynamic_range": round(dynamic_range, 4),
            "active_ratio": round(active_ratio, 4),
        },
    )


def transcribe_with_basic_pitch(input_path: Path, output_dir: Path) -> tuple[Path, int]:
    os.environ.setdefault("TMPDIR", "/private/tmp")
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/soundclay-pycache")

    with contextlib.redirect_stdout(sys.stderr):
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except ImportError as error:
            raise RuntimeError(
                "Basic Pitch is not installed. Run the app with the project .venv or install basic-pitch."
            ) from error

        _, midi_data, note_events = predict(
            input_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=90,
            midi_tempo=120,
        )

    midi_path = output_dir / f"{input_path.stem}.mid"
    midi_data.write(str(midi_path))
    return midi_path, len(note_events)


def process(input_path: Path, output_dir: Path) -> dict[str, object]:
    audio = read_wav(input_path)
    frames = analyze_frames(audio)
    instrument, confidence, features = classify_instrument(input_path, frames)
    model = MODEL_BY_INSTRUMENT[instrument]

    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path, note_count = transcribe_with_basic_pitch(input_path, output_dir)

    if note_count == 0:
        raise ValueError("Basic Pitch did not detect notes. Try a cleaner single-instrument WAV file.")

    return {
        "instrument": instrument,
        "confidence": round(confidence, 2),
        "model": model,
        "midiPath": str(midi_path),
        "noteCount": note_count,
        "durationSeconds": round(len(audio.samples) / audio.sample_rate, 2),
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        result = process(Path(args.input), Path(args.output_dir))
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
