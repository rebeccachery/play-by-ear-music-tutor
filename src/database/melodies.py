import numpy as np

# Melodies library
MELODIES = [
    {
        "id": "c_major_scale",
        "name": "C Major Scale (Ascending)",
        "notes": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
        "durations": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "description": "The foundation of Western music. Play 8 notes ascending steadily.",
        "difficulty": "Beginner"
    },
    {
        "id": "ode_to_joy",
        "name": "Ode to Joy (Opening)",
        "notes": ["E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4", "C4", "C4", "D4", "E4", "E4", "D4", "D4"],
        "durations": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 0.5, 2.0],
        "description": "Beethoven's famous theme. Pay attention to the rhythm at the end of the phrase!",
        "difficulty": "Easy"
    },
    {
        "id": "twinkle_twinkle",
        "name": "Twinkle Twinkle Little Star (Theme)",
        "notes": ["C4", "C4", "G4", "G4", "A4", "A4", "G4", "F4", "F4", "E4", "E4", "D4", "D4", "C4"],
        "durations": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
        "description": "A classic nursery rhyme. Steady quarter notes with a half note pause on the G4 and final C4.",
        "difficulty": "Easy"
    },
    {
        "id": "c_major_arpeggio",
        "name": "C Major Arpeggio",
        "notes": ["C4", "E4", "G4", "C5", "G4", "E4", "C4"],
        "durations": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        "description": "Play the notes of the C Major chord individually, ascending and descending.",
        "difficulty": "Intermediate"
    }
]

def get_melody_by_id(melody_id):
    for m in MELODIES:
        if m["id"] == melody_id:
            return m
    return None
