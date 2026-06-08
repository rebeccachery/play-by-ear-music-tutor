"""Backward-compatible melody exports. Prefer src.database.drills for new code."""

from src.database.drills import MELODY_DRILLS, get_drill_by_id

MELODIES = MELODY_DRILLS


def get_melody_by_id(melody_id: str):
    return get_drill_by_id(melody_id)
