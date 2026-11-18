"""End-to-end check: the app's own reference audio should score (near) perfectly.

This catches regressions in note naming, note segmentation and scoring weights,
since a clean synthesized performance of the drill is the best possible input.
"""
import pytest
import soundfile as sf

from src.audio_processing.librosa_pyin import LibrosaPyinTracker
from src.audio_processing.synth import synthesize_drill
from src.database.drills import DRILLS
from src.evaluation.melody_aligner import MelodyAligner


@pytest.fixture(scope="module")
def tracker():
    return LibrosaPyinTracker()


@pytest.mark.parametrize("drill", DRILLS, ids=[d["id"] for d in DRILLS])
def test_reference_audio_scores_perfectly(drill, tracker, tmp_path):
    path = tmp_path / "ref.wav"
    sf.write(path, synthesize_drill(drill), 22050)

    detected = tracker.transcribe(str(path))
    result = MelodyAligner().align(
        drill["notes"],
        detected,
        drill_type=drill["type"],
        target_durations=drill["durations"],
        tempo_bpm=drill["tempo_bpm"],
    )

    kinds = [step["type"] for step in result["alignment"]]
    assert kinds == ["match"] * len(drill["notes"]), (
        f"detected {[n['note'] for n in detected]} for target {drill['notes']}"
    )
    assert result["pitch_score"] == pytest.approx(1.0)
    assert result["accuracy"] >= 0.95


def test_sustained_note_with_vibrato_is_one_note(tracker, tmp_path):
    """Re-attack splitting must not chop a sung note with vibrato/tremolo."""
    import numpy as np

    sr = 22050
    t = np.arange(int(sr * 3)) / sr
    rng = np.random.default_rng(0)
    freq = 440 * 2 ** ((0.3 * np.sin(2 * np.pi * 5.5 * t)) / 12)  # ±30 cents
    phase = 2 * np.pi * np.cumsum(freq) / sr
    y = (np.sin(phase) + 0.3 * np.sin(2 * phase)) * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * t))
    y += 0.02 * rng.standard_normal(len(t))
    path = tmp_path / "vibrato.wav"
    sf.write(path, (y / 2).astype(np.float32), sr)

    assert [n["note"] for n in tracker.transcribe(str(path))] == ["A4"]
