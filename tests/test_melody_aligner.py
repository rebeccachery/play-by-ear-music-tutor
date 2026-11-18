import pytest

from src.audio_processing.librosa_pyin import midi_to_note_name
from src.evaluation.melody_aligner import MelodyAligner, note_to_midi


def played(notes, starts=None, length=0.9):
    starts = starts or [float(i) for i in range(len(notes))]
    return [
        {"note": n, "frequency": 0.0, "start_time": s, "end_time": s + length}
        for n, s in zip(notes, starts)
    ]


@pytest.mark.parametrize("spellings", [("C#4", "Db4", "C♯4"), ("A#4", "Bb4", "B♭4")])
def test_enharmonic_spellings_are_equal(spellings):
    assert len({note_to_midi(n) for n in spellings}) == 1


def test_tracker_names_are_ascii():
    assert midi_to_note_name(61) == "C#4"


def test_flat_target_matches_sharp_detection():
    r = MelodyAligner().align(["C4", "Db4"], played(["C4", "C#4"]), drill_type="interval")
    assert [s["type"] for s in r["alignment"]] == ["match", "match"]
    assert r["accuracy"] == pytest.approx(1.0)


def test_perfect_interval_is_100_percent():
    r = MelodyAligner().align(["C4", "G4"], played(["C4", "G4"]), drill_type="interval")
    assert r["interval_score"] == 1.0
    assert r["accuracy"] == pytest.approx(1.0)


def test_interval_ignores_stray_leading_note():
    r = MelodyAligner().align(["C4", "G4"], played(["F2", "C4", "G4"]), drill_type="interval")
    assert r["interval_score"] == 1.0


def test_octave_error_gets_partial_credit():
    r = MelodyAligner().align(["C4"], played(["C5"]), drill_type="single_note")
    assert r["accuracy"] == 0.5
    assert "wrong octave" in r["tips"][0]


def test_written_rhythm_is_not_penalised():
    # Quarter, quarter, half, quarter at 60 bpm -> starts at 0, 1, 2, 4
    r = MelodyAligner().align(
        ["C4", "E4", "G4", "C5"],
        played(["C4", "E4", "G4", "C5"], starts=[0.0, 1.0, 2.0, 4.0]),
        drill_type="melody",
        target_durations=[1.0, 1.0, 2.0, 1.0],
        tempo_bpm=60,
    )
    assert r["timing_score"] == pytest.approx(1.0)
    assert "great" in r["timing_feedback"]


def test_rushing_detected_from_tempo():
    r = MelodyAligner().align(
        ["C4", "D4", "E4", "F4"],
        played(["C4", "D4", "E4", "F4"], starts=[0.0, 0.5, 1.0, 1.5], length=0.4),
        drill_type="melody",
        target_durations=[1.0] * 4,
        tempo_bpm=60,
    )
    assert "rushing" in r["timing_feedback"]


def test_extra_note_is_insertion():
    r = MelodyAligner().align(["C4", "D4", "E4", "F4"], played(["C4", "A4", "D4", "E4", "F4"]), drill_type="melody")
    assert [s["type"] for s in r["alignment"]] == ["match", "insertion", "match", "match", "match"]


def test_skipped_note_is_deletion():
    r = MelodyAligner().align(["C4", "D4", "E4", "F4"], played(["C4", "E4", "F4"]), drill_type="melody")
    assert [s["type"] for s in r["alignment"]] == ["match", "deletion", "match", "match"]
