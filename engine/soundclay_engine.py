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
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional


SUPPORTED_EXTENSIONS = {".wav"}
MODEL_NAME = "basic-pitch-icassp-2022"
MAX_SAMPLES = 8
MIN_SAMPLE_DURATION = 0.35

@dataclass
class AudioData:
    sample_rate: int
    samples: list[float]


def read_wav(path: Path) -> AudioData:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("v1.2 currently supports WAV input. Please export your audio as .wav.")

    import numpy as np
    import soundfile as sf

    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(values, axis=1)
    return AudioData(sample_rate=sample_rate, samples=mono.tolist())


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


def transcribe_with_basic_pitch(
    input_path: Path, output_dir: Path
) -> tuple[Path, list[tuple[float, float, int, float, Optional[list[int]]]]]:
    for variable in ("TMPDIR", "TMP", "TEMP"):
        os.environ[variable] = "/private/tmp"
    os.environ["PYTHONPYCACHEPREFIX"] = "/private/tmp/soundclay-pycache"
    tempfile.tempdir = "/private/tmp"

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
    return midi_path, note_events


def note_overlaps(
    candidate: tuple[float, float, int, float, Optional[list[int]]],
    note_events: list[tuple[float, float, int, float, Optional[list[int]]]],
) -> bool:
    start, end, *_ = candidate
    for other in note_events:
        if other is candidate:
            continue
        other_start, other_end, *_ = other
        if min(end, other_end) - max(start, other_start) > 0.04:
            return True
    return False


def sample_candidates(
    note_events: list[tuple[float, float, int, float, Optional[list[int]]]],
) -> list[tuple[float, float, int, float, Optional[list[int]]]]:
    isolated = [
        event
        for event in note_events
        if event[1] - event[0] >= MIN_SAMPLE_DURATION and not note_overlaps(event, note_events)
    ]
    pool = isolated or [
        event for event in note_events if event[1] - event[0] >= MIN_SAMPLE_DURATION
    ]

    best_by_pitch: dict[int, tuple[float, float, int, float, Optional[list[int]]]] = {}
    for event in pool:
        start, end, pitch, amplitude, _ = event
        duration = min(end - start, 2.5)
        score = duration * 0.65 + amplitude * 0.35
        existing = best_by_pitch.get(pitch)
        if existing is None:
            best_by_pitch[pitch] = event
            continue
        existing_score = min(existing[1] - existing[0], 2.5) * 0.65 + existing[3] * 0.35
        if score > existing_score:
            best_by_pitch[pitch] = event

    ranked = sorted(
        best_by_pitch.values(),
        key=lambda event: (min(event[1] - event[0], 2.5) * 0.65 + event[3] * 0.35),
        reverse=True,
    )
    selected: list[tuple[float, float, int, float, Optional[list[int]]]] = []
    for event in ranked:
        if all(abs(event[2] - chosen[2]) >= 3 for chosen in selected):
            selected.append(event)
        if len(selected) == MAX_SAMPLES:
            break

    return sorted(selected, key=lambda event: event[2])


def midi_note_name(note: int) -> str:
    names = ("C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B")
    return f"{names[note % 12]}{note // 12 - 1}"


def write_sample(
    samples: list[float],
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
) -> None:
    import numpy as np
    import soundfile as sf

    start = max(0, int((start_seconds - 0.025) * sample_rate))
    end = min(len(samples), int((end_seconds + 0.12) * sample_rate))
    audio = np.asarray(samples[start:end], dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Could not extract audio for an instrument sample.")

    audio -= np.mean(audio)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= min(4.0, 0.95 / peak)

    fade_in = min(audio.size // 4, max(1, int(sample_rate * 0.005)))
    fade_out = min(audio.size // 3, max(1, int(sample_rate * 0.04)))
    audio[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    audio[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")


def sfz_key_ranges(pitches: list[int]) -> list[tuple[int, int]]:
    if len(pitches) == 1:
        return [(0, 127)]

    ranges = []
    for index, pitch in enumerate(pitches):
        low = 0 if index == 0 else (pitches[index - 1] + pitch) // 2 + 1
        high = 127 if index == len(pitches) - 1 else (pitch + pitches[index + 1]) // 2
        ranges.append((low, high))
    return ranges


def build_instrument_package(
    input_path: Path,
    output_dir: Path,
    midi_path: Path,
    note_events: list[tuple[float, float, int, float, Optional[list[int]]]],
    audio: AudioData,
) -> tuple[Path, Path, int]:
    candidates = sample_candidates(note_events)
    if not candidates:
        raise ValueError(
            "No notes were long enough to build an SFZ instrument. "
            "Try audio with notes held for at least 0.35 seconds."
        )

    instrument_dir = output_dir / f"{input_path.stem}_instrument"
    samples_dir = instrument_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    sample_files = []
    for index, event in enumerate(candidates, start=1):
        start, end, pitch, *_ = event
        sample_name = f"{index:02d}_{midi_note_name(pitch)}.wav"
        write_sample(
            audio.samples,
            audio.sample_rate,
            start,
            end,
            samples_dir / sample_name,
        )
        sample_files.append((pitch, sample_name))

    sfz_path = instrument_dir / f"{input_path.stem}.sfz"
    pitches = [pitch for pitch, _ in sample_files]
    lines = [
        "// Generated by SoundClay v1.2",
        "<global> ampeg_attack=0.005 ampeg_release=0.25",
        "",
    ]
    for (pitch, sample_name), (low, high) in zip(sample_files, sfz_key_ranges(pitches)):
        lines.append(
            f"<region> sample=samples/{sample_name} "
            f"pitch_keycenter={pitch} lokey={low} hikey={high}"
        )
    sfz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    package_path = output_dir / f"{input_path.stem}_soundclay.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(midi_path, f"{input_path.stem}.mid")
        archive.write(sfz_path, sfz_path.name)
        for _, sample_name in sample_files:
            archive.write(samples_dir / sample_name, f"samples/{sample_name}")

    return sfz_path, package_path, len(sample_files)


def process(input_path: Path, output_dir: Path) -> dict[str, object]:
    audio = read_wav(input_path)
    frames = analyze_frames(audio)
    instrument, confidence, features = classify_instrument(input_path, frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path, note_events = transcribe_with_basic_pitch(input_path, output_dir)
    note_count = len(note_events)

    if note_count == 0:
        raise ValueError("Basic Pitch did not detect notes. Try a cleaner single-instrument WAV file.")

    sfz_path, package_path, sample_count = build_instrument_package(
        input_path,
        output_dir,
        midi_path,
        note_events,
        audio,
    )

    return {
        "instrument": instrument,
        "confidence": round(confidence, 2),
        "model": MODEL_NAME,
        "midiPath": str(midi_path),
        "sfzPath": str(sfz_path),
        "packagePath": str(package_path),
        "noteCount": note_count,
        "sampleCount": sample_count,
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
