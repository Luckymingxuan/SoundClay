#!/usr/bin/env python3
"""SoundClay v1.1 audio-to-MIDI engine.

This first backend is intentionally dependency-free. It provides the same
pipeline shape that model-backed transcribers will use later:
classify instrument -> choose transcription model -> emit MIDI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional


SUPPORTED_EXTENSIONS = {".wav"}

MODEL_BY_INSTRUMENT = {
    "Piano": "heuristic-piano-transcriber",
    "Guitar": "heuristic-plucked-string-transcriber",
    "Bass": "heuristic-bass-transcriber",
    "Vocal": "heuristic-vocal-melody-transcriber",
    "Synth": "heuristic-synth-transcriber",
    "Strings": "heuristic-sustained-strings-transcriber",
}

MIDI_PROGRAM_BY_INSTRUMENT = {
    "Piano": 0,
    "Guitar": 24,
    "Bass": 33,
    "Vocal": 53,
    "Synth": 80,
    "Strings": 48,
}


@dataclass
class AudioData:
    sample_rate: int
    samples: list[float]


@dataclass
class Note:
    midi: int
    start: float
    duration: float
    velocity: int


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


def frequency_to_midi(frequency: float) -> int:
    return max(21, min(108, round(69 + 12 * math.log2(frequency / 440.0))))


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


def transcribe_notes(audio: AudioData, frames: list[dict[str, float | None]]) -> list[Note]:
    active_rms = [float(frame["rms"] or 0) for frame in frames if (frame["rms"] or 0) > 0.02]
    threshold = max(0.025, median(active_rms) * 0.7) if active_rms else 0.025

    notes: list[Note] = []
    current_midi: int | None = None
    current_start = 0.0
    current_velocity = 64

    for frame in frames:
        pitch = frame["pitch"]
        level = float(frame["rms"] or 0)
        time = float(frame["time"] or 0)
        midi = frequency_to_midi(pitch) if pitch and level >= threshold else None

        if midi is None:
            if current_midi is not None:
                duration = max(0.08, time - current_start)
                notes.append(Note(current_midi, current_start, duration, current_velocity))
                current_midi = None
            continue

        velocity = max(35, min(118, int(36 + level * 420)))
        if current_midi is None:
            current_midi = midi
            current_start = time
            current_velocity = velocity
        elif abs(midi - current_midi) > 1:
            duration = max(0.08, time - current_start)
            notes.append(Note(current_midi, current_start, duration, current_velocity))
            current_midi = midi
            current_start = time
            current_velocity = velocity
        else:
            current_velocity = max(current_velocity, velocity)

    if current_midi is not None:
        final_time = len(audio.samples) / audio.sample_rate
        notes.append(Note(current_midi, current_start, max(0.08, final_time - current_start), current_velocity))

    return merge_short_notes(notes)


def merge_short_notes(notes: list[Note]) -> list[Note]:
    merged: list[Note] = []
    for note in notes:
        if merged and note.midi == merged[-1].midi and note.start - (merged[-1].start + merged[-1].duration) < 0.09:
            previous = merged[-1]
            previous.duration = max(previous.duration, note.start + note.duration - previous.start)
            previous.velocity = max(previous.velocity, note.velocity)
        elif note.duration >= 0.07:
            merged.append(note)
    return merged


def vlq(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7

    output = bytearray()
    while True:
        output.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(output)


def write_midi(path: Path, notes: list[Note], instrument: str) -> None:
    ticks_per_quarter = 480
    tempo_microseconds = 500000
    program = MIDI_PROGRAM_BY_INSTRUMENT.get(instrument, 0)

    events: list[tuple[int, bytes]] = [
        (0, b"\xff\x51\x03" + tempo_microseconds.to_bytes(3, "big")),
        (0, b"\xc0" + bytes([program])),
    ]

    for note in notes:
        start_tick = round(note.start * 2 * ticks_per_quarter)
        end_tick = round((note.start + note.duration) * 2 * ticks_per_quarter)
        events.append((start_tick, b"\x90" + bytes([note.midi, note.velocity])))
        events.append((max(start_tick + 1, end_tick), b"\x80" + bytes([note.midi, 0])))

    events.sort(key=lambda item: (item[0], item[1][0] == 0x90))
    track = bytearray()
    previous_tick = 0
    for tick, payload in events:
        track.extend(vlq(max(0, tick - previous_tick)))
        track.extend(payload)
        previous_tick = tick
    track.extend(vlq(0))
    track.extend(b"\xff\x2f\x00")

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_quarter)
    track_chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    path.write_bytes(header + track_chunk)


def process(input_path: Path, output_dir: Path) -> dict[str, object]:
    audio = read_wav(input_path)
    frames = analyze_frames(audio)
    instrument, confidence, features = classify_instrument(input_path, frames)
    model = MODEL_BY_INSTRUMENT[instrument]
    notes = transcribe_notes(audio, frames)

    if not notes:
        raise ValueError("No confident notes were detected. Try a cleaner single-instrument WAV file.")

    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = output_dir / f"{input_path.stem}.mid"
    write_midi(midi_path, notes, instrument)

    return {
        "instrument": instrument,
        "confidence": round(confidence, 2),
        "model": model,
        "midiPath": str(midi_path),
        "noteCount": len(notes),
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
