import librosa
import numpy as np
from typing import List, Dict, Any, Set
from .base import BasePitchTracker

HOP_LENGTH = 512


def midi_to_note_name(midi: int) -> str:
    """ASCII note name with sharps (e.g. 61 -> 'C#4').

    librosa's default uses Unicode sharps ('C♯4'), which never matched the
    'C#4' / 'Db4' spellings used by the drill library.
    """
    return librosa.midi_to_note(int(midi), unicode=False)


class LibrosaPyinTracker(BasePitchTracker):
    """
    Pitch tracker implementation using Librosa's pYIN algorithm.
    Suitable for monophonic audio signals (singing, solo instrument plays).
    """
    def __init__(
        self,
        fmin: str = "C2",
        fmax: str = "C7",
        min_note_duration: float = 0.15,
        reattack_energy_ratio: float = 4.0,
        max_duration: float = 60.0,
    ):
        self.fmin_hz = librosa.note_to_hz(fmin)
        self.fmax_hz = librosa.note_to_hz(fmax)
        self.min_note_duration = min_note_duration
        # A same-pitch onset only splits a note if energy jumps by this factor,
        # so vibrato or bow/breath noise in a sustained note doesn't split it.
        self.reattack_energy_ratio = reattack_energy_ratio
        self.max_duration = max_duration

    def _reattack_frames(self, y: np.ndarray, sr: int) -> Set[int]:
        """Frames where a new note is struck (used to split repeated pitches)."""
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=HOP_LENGTH, backtrack=True
        )
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        frames = set()
        for f in onsets:
            before = rms[max(0, f - 4): f + 1]
            after = rms[f: f + 6]
            if len(before) and len(after) and after.max() >= self.reattack_energy_ratio * max(before.min(), 1e-6):
                frames.add(int(f))
        return frames

    def transcribe(self, audio_path) -> List[Dict[str, Any]]:
        """Transcribe a path or file-like object into note segments."""
        # Load audio (downsample to 22050 Hz for standard speech/music analysis)
        y, sr = librosa.load(audio_path, sr=22050, duration=self.max_duration)
        if len(y) == 0:
            return []

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin_hz,
            fmax=self.fmax_hz,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
        frame_duration = HOP_LENGTH / sr
        reattacks = self._reattack_frames(y, sr)

        notes_segments: List[Dict[str, Any]] = []
        current_midi = None
        current_freqs: List[float] = []
        start_time = None

        def close(end_time: float) -> None:
            if current_midi is not None and end_time - start_time >= self.min_note_duration:
                notes_segments.append({
                    "note": midi_to_note_name(current_midi),
                    "midi": int(current_midi),
                    "frequency": float(np.median(current_freqs)),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                })

        for i, freq in enumerate(f0):
            voiced = bool(voiced_flag[i]) and freq is not None and not np.isnan(freq)
            if not voiced:
                close(times[i])
                current_midi, current_freqs, start_time = None, [], None
                continue

            midi = int(np.round(librosa.hz_to_midi(freq)))
            same_note = midi == current_midi
            long_enough = start_time is not None and times[i] - start_time >= self.min_note_duration
            if same_note and not (i in reattacks and long_enough):
                current_freqs.append(freq)
                continue

            # New pitch, or the same pitch struck again.
            close(times[i])
            current_midi, current_freqs, start_time = midi, [freq], times[i]

        close(times[-1] + frame_duration)
        return notes_segments
