from typing import List, Dict, Any, Optional
import numpy as np

NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def note_to_midi(note: str) -> int:
    """Convert note name like 'C4' or 'Db4' to MIDI number."""
    pitch = note[:-1]
    octave = int(note[-1])
    return NOTE_TO_MIDI[pitch] + (octave + 1) * 12


def parse_note(note: str) -> tuple:
    """Return (pitch_class, octave) from a note name."""
    return note[:-1], int(note[-1])


class MelodyAligner:
    """
    Aligns the played note sequence with the expected/target note sequence
    and generates detailed musical feedback on pitch accuracy and timing.
    """
    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty

    def align(
        self,
        target_notes: List[str],
        played_notes: List[Dict[str, Any]],
        drill_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Align target_notes (e.g., ['C4', 'E4', 'G4']) with played_notes dictionaries.
        Returns alignment score, detailed alignment path, and feedback.

        drill_type adjusts scoring weights: single_note, interval, motif, melody.
        """
        N = len(target_notes)
        M = len(played_notes)
        
        # DP table initialization
        dp = np.zeros((N + 1, M + 1))
        for i in range(N + 1):
            dp[i][0] = i * self.gap_penalty
        for j in range(M + 1):
            dp[0][j] = j * self.gap_penalty
            
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                target_note = target_notes[i - 1]
                played_note = played_notes[j - 1]["note"]
                
                # Check match (exact match gives full score, octave mismatch gets partial, others get mismatch)
                if target_note == played_note:
                    score = self.match_score
                elif target_note[:-1] == played_note[:-1]: # same pitch class, wrong octave
                    score = self.mismatch_score + 1
                else:
                    score = self.mismatch_score
                    
                match = dp[i - 1][j - 1] + score
                delete = dp[i - 1][j] + self.gap_penalty
                insert = dp[i][j - 1] + self.gap_penalty
                dp[i][j] = max(match, delete, insert)
                
        # Backtracking to reconstruct alignment
        alignment = []
        i, j = N, M
        
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                target_note = target_notes[i - 1]
                played_info = played_notes[j - 1]
                played_note = played_info["note"]
                
                if target_note == played_note:
                    score = self.match_score
                elif target_note[:-1] == played_note[:-1]:
                    score = self.mismatch_score + 1
                else:
                    score = self.mismatch_score
                    
                if dp[i][j] == dp[i - 1][j - 1] + score:
                    status = "correct" if target_note == played_note else "incorrect"
                    alignment.append({
                        "type": "match" if status == "correct" else "substitution",
                        "target_note": target_note,
                        "played_note": played_note,
                        "played_info": played_info,
                        "feedback": f"Played {played_note} (expected {target_note})" if status == "incorrect" else "Correct!"
                    })
                    i -= 1
                    j -= 1
                    continue
            
            if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + self.gap_penalty):
                alignment.append({
                    "type": "deletion",
                    "target_note": target_notes[i - 1],
                    "played_note": None,
                    "played_info": None,
                    "feedback": f"Missed {target_notes[i - 1]}"
                })
                i -= 1
            else:
                alignment.append({
                    "type": "insertion",
                    "target_note": None,
                    "played_note": played_notes[j - 1]["note"],
                    "played_info": played_notes[j - 1],
                    "feedback": f"Extra note {played_notes[j - 1]['note']} played"
                })
                j -= 1
                
        alignment.reverse()
        
        scoring = self._score_alignment(alignment, target_notes, played_notes, drill_type)

        tips = self._generate_tips(alignment, drill_type, scoring)

        return {
            "alignment": alignment,
            "accuracy": scoring["accuracy"],
            "pitch_score": scoring["pitch_score"],
            "interval_score": scoring.get("interval_score"),
            "timing_score": scoring.get("timing_score"),
            "timing_feedback": scoring["timing_feedback"],
            "tips": tips,
            "drill_type": drill_type or "melody",
        }

    def _score_alignment(
        self,
        alignment: List[Dict[str, Any]],
        target_notes: List[str],
        played_notes: List[Dict[str, Any]],
        drill_type: Optional[str],
    ) -> Dict[str, Any]:
        N = len(target_notes)
        correct_count = sum(1 for step in alignment if step["type"] == "match")
        pitch_score = (correct_count / N) if N > 0 else 0.0

        partial_count = sum(
            1 for step in alignment
            if step["type"] == "substitution"
            and step.get("played_note")
            and parse_note(step["target_note"])[0] == parse_note(step["played_note"])[0]
        )
        if drill_type == "single_note" and N == 1:
            pitch_score = self._score_single_note(alignment, target_notes[0], played_notes)
            timing_feedback = "Rhythm is not scored for single-note drills."
            return {
                "accuracy": pitch_score,
                "pitch_score": pitch_score,
                "timing_feedback": timing_feedback,
            }

        interval_score = None
        if drill_type == "interval" and N >= 2:
            interval_score = self._score_interval(target_notes, played_notes)
            pitch_score = 0.7 * pitch_score + 0.3 * (partial_count / N if N else 0)
            accuracy = 0.6 * pitch_score + 0.4 * interval_score
            timing_feedback = "Focus on pitch for now — rhythm is secondary for intervals."
        elif drill_type in ("single_note", "interval"):
            accuracy = pitch_score
            timing_feedback = (
                "Rhythm is not scored for this drill type."
                if drill_type == "interval"
                else "Rhythm is not scored for single-note drills."
            )
        elif drill_type == "motif":
            accuracy = 0.85 * pitch_score + 0.15 * self._rhythm_consistency_score(alignment)
            timing_feedback = self._evaluate_timing(alignment, include_rhythm_score=False)
        else:
            rhythm_score = self._rhythm_consistency_score(alignment)
            accuracy = 0.7 * pitch_score + 0.3 * rhythm_score
            timing_feedback = self._evaluate_timing(alignment, include_rhythm_score=False)

        result: Dict[str, Any] = {
            "accuracy": accuracy,
            "pitch_score": pitch_score,
            "timing_feedback": timing_feedback,
        }
        if interval_score is not None:
            result["interval_score"] = interval_score
        if drill_type in ("motif", "melody", None):
            result["timing_score"] = self._rhythm_consistency_score(alignment)
        return result

    def _score_single_note(
        self,
        alignment: List[Dict[str, Any]],
        target_note: str,
        played_notes: List[Dict[str, Any]],
    ) -> float:
        for step in alignment:
            if step["type"] == "match" and step["target_note"] == target_note:
                return 1.0
            if step["type"] == "substitution" and step.get("played_note"):
                if parse_note(step["target_note"])[0] == parse_note(step["played_note"])[0]:
                    return 0.5
        if played_notes:
            played = played_notes[0]["note"]
            if played == target_note:
                return 1.0
            if parse_note(played)[0] == parse_note(target_note)[0]:
                return 0.5
        return 0.0

    def _score_interval(
        self,
        target_notes: List[str],
        played_notes: List[Dict[str, Any]],
    ) -> float:
        if len(target_notes) < 2 or len(played_notes) < 2:
            return 0.0
        target_semitones = note_to_midi(target_notes[1]) - note_to_midi(target_notes[0])
        played_semitones = note_to_midi(played_notes[1]["note"]) - note_to_midi(played_notes[0]["note"])
        return 1.0 if target_semitones == played_semitones else 0.0

    def _rhythm_consistency_score(self, alignment: List[Dict[str, Any]]) -> float:
        played_durations = []
        for step in alignment:
            if step["type"] == "match" and step["played_info"] is not None:
                info = step["played_info"]
                played_durations.append(info["end_time"] - info["start_time"])
        if len(played_durations) < 2:
            return 0.5
        mean_duration = np.mean(played_durations)
        std_duration = np.std(played_durations)
        coef_var = std_duration / mean_duration if mean_duration > 0 else 1.0
        return float(max(0.0, 1.0 - coef_var))

    def _generate_tips(
        self,
        alignment: List[Dict[str, Any]],
        drill_type: Optional[str],
        scoring: Dict[str, Any],
    ) -> List[str]:
        tips = []
        for step in alignment:
            if step["type"] == "substitution":
                target = step["target_note"]
                played = step["played_note"]
                if drill_type == "single_note":
                    tips.append(f"You played {played} — the target was {target}.")
                else:
                    tips.append(
                        f"Work on the transition to {target} (you played {played} instead)."
                    )
            elif step["type"] == "deletion":
                tips.append(f"Make sure to play {step['target_note']}; it was missed.")
            elif step["type"] == "insertion":
                tips.append(f"Extra note {step['played_note']} — try to play only what you hear.")

        if drill_type == "interval" and scoring.get("interval_score") == 0.0:
            tips.append("The interval size was off — listen for the distance between the two notes.")

        if not tips:
            tips.append("Excellent work! You played the drill correctly.")
        return list(dict.fromkeys(tips))[:3]

    def _evaluate_timing(
        self,
        alignment: List[Dict[str, Any]],
        include_rhythm_score: bool = True,
    ) -> str:
        """Evaluate user rhythm and speed based on matched notes."""
        played_durations = []
        
        for step in alignment:
            if step["type"] == "match" and step["played_info"] is not None:
                info = step["played_info"]
                played_dur_val = info["end_time"] - info["start_time"]
                played_durations.append(played_dur_val)
                
        if len(played_durations) < 2:
            return "Unable to evaluate timing. Try playing with a steadier tempo."
            
        # Calculate consistency (variance of durations for a simple steady exercise)
        mean_duration = np.mean(played_durations)
        std_duration = np.std(played_durations)
        coef_var = std_duration / mean_duration if mean_duration > 0 else 0
        
        # Simple tempo heuristic
        # If notes are generally very short (< 0.25s average), they might be rushing
        if mean_duration < 0.25:
            speed_desc = "rushing (tempo is very fast)."
        elif mean_duration > 1.2:
            speed_desc = "dragging (tempo is very slow)."
        else:
            speed_desc = "steady"
            
        if coef_var > 0.4:
            return f"Your rhythm is a bit uneven (coefficient of variation: {coef_var:.2f}). Try practicing with a metronome."
        else:
            if speed_desc == "steady":
                return "Your timing and rhythm are great! Nice, steady play."
            else:
                return f"Your rhythm is stable, but you are {speed_desc} Try adjusting your overall pace."
