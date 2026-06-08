"""
Ear Training Fundamentals drill library.

Drill types (in progression order):
  - single_note: hear one pitch, reproduce it
  - interval:    hear two notes, reproduce the interval
  - motif:       short 3–5 note pattern
  - melody:      full phrase with rhythm
"""

from typing import List, Dict, Any, Optional

DRILL_TYPES = ("single_note", "interval", "motif", "melody")

DRILL_LEVELS = {
    1: "Single Notes",
    2: "Intervals",
    3: "Short Motifs",
    4: "Melodies",
}


def _drill(
    id: str,
    name: str,
    drill_type: str,
    level: int,
    notes: List[str],
    durations: List[float],
    description: str,
    difficulty: str = "Beginner",
    tempo_bpm: int = 60,
    tags: Optional[List[str]] = None,
    interval_name: Optional[str] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "id": id,
        "name": name,
        "type": drill_type,
        "level": level,
        "notes": notes,
        "durations": durations,
        "description": description,
        "difficulty": difficulty,
        "tempo_bpm": tempo_bpm,
        "tags": tags or [],
    }
    if interval_name:
        d["interval_name"] = interval_name
    return d


# ---------------------------------------------------------------------------
# Level 1 — Single Notes
# ---------------------------------------------------------------------------
SINGLE_NOTE_DRILLS = [
    _drill(
        "sn_c4", "Middle C", "single_note", 1,
        ["C4"], [2.0],
        "Listen to Middle C, then play the same note on your instrument.",
        tags=["white key", "landmark"],
    ),
    _drill(
        "sn_d4", "D", "single_note", 1,
        ["D4"], [2.0],
        "Identify and play D — the white key between the two black keys.",
    ),
    _drill(
        "sn_e4", "E", "single_note", 1,
        ["E4"], [2.0],
        "Identify and play E — the white key to the right of the two black keys.",
    ),
    _drill(
        "sn_f4", "F", "single_note", 1,
        ["F4"], [2.0],
        "Identify and play F — the white key left of the group of three black keys.",
    ),
    _drill(
        "sn_g4", "G", "single_note", 1,
        ["G4"], [2.0],
        "Identify and play G — the first white key inside the three black keys.",
    ),
    _drill(
        "sn_a4", "A", "single_note", 1,
        ["A4"], [2.0],
        "Identify and play A — concert pitch reference (440 Hz).",
        tags=["reference pitch"],
    ),
    _drill(
        "sn_b4", "B", "single_note", 1,
        ["B4"], [2.0],
        "Identify and play B — the white key right of the three black keys.",
    ),
    _drill(
        "sn_c5", "High C", "single_note", 1,
        ["C5"], [2.0],
        "Identify and play C one octave above Middle C.",
        tags=["octave"],
    ),
]

# ---------------------------------------------------------------------------
# Level 2 — Intervals (ascending from C4 unless noted)
# ---------------------------------------------------------------------------
INTERVAL_DRILLS = [
    _drill(
        "int_m2_c_d", "Minor 2nd (C → D♭)", "interval", 2,
        ["C4", "Db4"], [1.5, 1.5],
        "Hear a minor second, then play both notes in order.",
        difficulty="Beginner",
        interval_name="Minor 2nd",
    ),
    _drill(
        "int_M2_c_d", "Major 2nd (C → D)", "interval", 2,
        ["C4", "D4"], [1.5, 1.5],
        "Hear a major second (whole step), then play both notes.",
        interval_name="Major 2nd",
    ),
    _drill(
        "int_m3_c_eb", "Minor 3rd (C → E♭)", "interval", 2,
        ["C4", "Eb4"], [1.5, 1.5],
        "Hear a minor third, then play both notes.",
        interval_name="Minor 3rd",
    ),
    _drill(
        "int_M3_c_e", "Major 3rd (C → E)", "interval", 2,
        ["C4", "E4"], [1.5, 1.5],
        "Hear a major third — the bright, happy interval.",
        interval_name="Major 3rd",
    ),
    _drill(
        "int_P4_c_f", "Perfect 4th (C → F)", "interval", 2,
        ["C4", "F4"], [1.5, 1.5],
        "Hear a perfect fourth, then play both notes.",
        interval_name="Perfect 4th",
    ),
    _drill(
        "int_tritone_c_fsharp", "Tritone (C → F♯)", "interval", 2,
        ["C4", "F#4"], [1.5, 1.5],
        "Hear the tritone — the most dissonant interval.",
        difficulty="Intermediate",
        interval_name="Tritone",
    ),
    _drill(
        "int_P5_c_g", "Perfect 5th (C → G)", "interval", 2,
        ["C4", "G4"], [1.5, 1.5],
        "Hear a perfect fifth — the strong, open interval.",
        interval_name="Perfect 5th",
    ),
    _drill(
        "int_m6_c_ab", "Minor 6th (C → A♭)", "interval", 2,
        ["C4", "Ab4"], [1.5, 1.5],
        "Hear a minor sixth, then play both notes.",
        difficulty="Intermediate",
        interval_name="Minor 6th",
    ),
    _drill(
        "int_M6_c_a", "Major 6th (C → A)", "interval", 2,
        ["C4", "A4"], [1.5, 1.5],
        "Hear a major sixth, then play both notes.",
        difficulty="Intermediate",
        interval_name="Major 6th",
    ),
    _drill(
        "int_m7_c_bb", "Minor 7th (C → B♭)", "interval", 2,
        ["C4", "Bb4"], [1.5, 1.5],
        "Hear a minor seventh, then play both notes.",
        difficulty="Intermediate",
        interval_name="Minor 7th",
    ),
    _drill(
        "int_M7_c_b", "Major 7th (C → B)", "interval", 2,
        ["C4", "B4"], [1.5, 1.5],
        "Hear a major seventh, then play both notes.",
        difficulty="Intermediate",
        interval_name="Major 7th",
    ),
    _drill(
        "int_P8_c_c5", "Octave (C → C)", "interval", 2,
        ["C4", "C5"], [1.5, 1.5],
        "Hear an octave leap, then play both notes.",
        interval_name="Octave",
    ),
]

# ---------------------------------------------------------------------------
# Level 3 — Short Motifs (3–5 notes)
# ---------------------------------------------------------------------------
MOTIF_DRILLS = [
    _drill(
        "motif_c_major_triad", "C Major Triad", "motif", 3,
        ["C4", "E4", "G4"], [1.0, 1.0, 2.0],
        "Play the three notes of a C major chord in order.",
        difficulty="Easy",
    ),
    _drill(
        "motif_stepwise_up", "Stepwise Ascending", "motif", 3,
        ["C4", "D4", "E4", "F4"], [1.0, 1.0, 1.0, 1.0],
        "Four notes stepping up — focus on smooth transitions.",
        difficulty="Easy",
    ),
    _drill(
        "motif_stepwise_down", "Stepwise Descending", "motif", 3,
        ["G4", "F4", "E4", "D4"], [1.0, 1.0, 1.0, 1.0],
        "Four notes stepping down from G.",
        difficulty="Easy",
    ),
    _drill(
        "motif_ode_opening", "Ode to Joy (Opening)", "motif", 3,
        ["E4", "E4", "F4", "G4"], [1.0, 1.0, 1.0, 1.0],
        "The first four notes of Beethoven's Ode to Joy.",
        difficulty="Easy",
    ),
    _drill(
        "motif_twinkle_opening", "Twinkle Opening", "motif", 3,
        ["C4", "C4", "G4", "G4"], [1.0, 1.0, 1.0, 1.0],
        "The opening phrase of Twinkle Twinkle Little Star.",
        difficulty="Easy",
    ),
    _drill(
        "motif_arpeggio_up", "C Major Arpeggio (Up)", "motif", 3,
        ["C4", "E4", "G4", "C5"], [1.0, 1.0, 1.0, 2.0],
        "Ascending C major arpeggio — stretch to the high C.",
        difficulty="Intermediate",
    ),
    _drill(
        "motif_leap_return", "Leap & Return", "motif", 3,
        ["C4", "G4", "C4"], [1.0, 1.0, 2.0],
        "Jump up a fifth to G, then return to C.",
        difficulty="Intermediate",
    ),
]

# ---------------------------------------------------------------------------
# Level 4 — Full Melodies
# ---------------------------------------------------------------------------
MELODY_DRILLS = [
    _drill(
        "mel_c_major_scale", "C Major Scale (Ascending)", "melody", 4,
        ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "The foundation of Western music. Play 8 notes ascending steadily.",
        difficulty="Beginner",
    ),
    _drill(
        "mel_ode_to_joy", "Ode to Joy (Opening)", "melody", 4,
        ["E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4", "C4", "C4", "D4", "E4", "E4", "D4", "D4"],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 0.5, 2.0],
        "Beethoven's famous theme. Pay attention to the rhythm at the end!",
        difficulty="Easy",
    ),
    _drill(
        "mel_twinkle", "Twinkle Twinkle Little Star", "melody", 4,
        ["C4", "C4", "G4", "G4", "A4", "A4", "G4", "F4", "F4", "E4", "E4", "D4", "D4", "C4"],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "Classic nursery rhyme with half-note holds on G and the final C.",
        difficulty="Easy",
    ),
    _drill(
        "mel_c_major_arpeggio", "C Major Arpeggio", "melody", 4,
        ["C4", "E4", "G4", "C5", "G4", "E4", "C4"],
        [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        "C major chord tones ascending and descending.",
        difficulty="Intermediate",
    ),
]

DRILLS: List[Dict[str, Any]] = (
    SINGLE_NOTE_DRILLS
    + INTERVAL_DRILLS
    + MOTIF_DRILLS
    + MELODY_DRILLS
)


def get_drill_by_id(drill_id: str) -> Optional[Dict[str, Any]]:
    for d in DRILLS:
        if d["id"] == drill_id:
            return d
    return None


def get_drills_by_level(level: int) -> List[Dict[str, Any]]:
    return [d for d in DRILLS if d["level"] == level]


def get_drills_by_type(drill_type: str) -> List[Dict[str, Any]]:
    return [d for d in DRILLS if d["type"] == drill_type]


def duration_to_seconds(duration: float, tempo_bpm: int = 60) -> float:
    """Convert beat-unit duration to seconds (1.0 = quarter note)."""
    return duration * (60.0 / tempo_bpm)
