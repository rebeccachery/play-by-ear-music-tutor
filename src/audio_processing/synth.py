"""Reference-tone synthesis for drills (kept free of Streamlit so it can be tested)."""

import io
from typing import Any, Dict

import librosa
import numpy as np
import soundfile as sf

from src.database.drills import duration_to_seconds

NOTE_GAP_SECONDS = 0.05


def note_to_hz(note: str) -> float:
    """Frequency for a note name such as 'C4', 'Db4', 'F#4' or 'C♯4'."""
    return float(librosa.note_to_hz(note))


def synthesize_drill(drill: Dict[str, Any], sample_rate: int = 22050) -> np.ndarray:
    """Render a drill as decaying sine tones separated by short silences."""
    tempo_bpm = drill.get("tempo_bpm", 60)
    chunks = []
    for note, dur in zip(drill["notes"], drill["durations"]):
        freq = note_to_hz(note)
        note_seconds = duration_to_seconds(dur, tempo_bpm)
        t = np.linspace(0, note_seconds, int(sample_rate * note_seconds), endpoint=False)
        fade = np.linspace(1.0, 0.01, len(t))
        chunks.append(np.sin(2 * np.pi * freq * t) * fade)
        chunks.append(np.zeros(int(sample_rate * NOTE_GAP_SECONDS)))
    return np.concatenate(chunks).astype(np.float32)


def generate_reference_audio(drill: Dict[str, Any], sample_rate: int = 22050) -> bytes:
    """Synthesize a drill and return it as WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, synthesize_drill(drill, sample_rate), sample_rate, format="wav")
    return buffer.getvalue()
