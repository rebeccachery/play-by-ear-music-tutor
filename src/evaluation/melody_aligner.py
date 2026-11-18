from typing import List, Dict, Any, Optional
import librosa
import numpy as np


def note_to_midi(note: str) -> int:
    """MIDI number for 'C4', 'Db4', 'C#4' or 'C♯4' (all spellings agree)."""
    return int(librosa.note_to_midi(note))


def played_midi(played: Dict[str, Any]) -> int:
    """MIDI number of a detected note dict (uses 'midi' if the tracker set it)."""
    if played.get("midi") is not None:
        return int(played["midi"])
    return note_to_midi(played["note"])


def same_pitch_class(a: int, b: int) -> bool:
    return a % 12 == b % 12


class MelodyAligner:
    """
    Aligns the played note sequence with the expected/target note sequence
    and generates detailed musical feedback on pitch accuracy and timing.
    """
    # Played tempo within these factors of the written tempo counts as on-tempo.
    RUSHING_FACTOR = 0.8
    DRAGGING_FACTOR = 1.25

    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty

    def _pair_score(self, target_midi: int, played: int) -> int:
        if target_midi == played:
            return self.match_score
        if same_pitch_class(target_midi, played):  # right note, wrong octave
            return self.mismatch_score + 1
        return self.mismatch_score

    def align(
        self,
        target_notes: List[str],
        played_notes: List[Dict[str, Any]],
        drill_type: Optional[str] = None,
        target_durations: Optional[List[float]] = None,
        tempo_bpm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Align target_notes (e.g., ['C4', 'E4', 'G4']) with played_notes dictionaries.
        Returns alignment score, detailed alignment path, and feedback.

        drill_type adjusts scoring weights: single_note, interval, motif, melody.
        target_durations (beats) and tempo_bpm enable rhythm/tempo scoring
        against the written rhythm; without them, equal beats are assumed.
        """
        N = len(target_notes)
        M = len(played_notes)
        target_midi = [note_to_midi(n) for n in target_notes]
        played_m = [played_midi(p) for p in played_notes]

        # DP table initialization
        dp = np.zeros((N + 1, M + 1))
        for i in range(N + 1):
            dp[i][0] = i * self.gap_penalty
        for j in range(M + 1):
            dp[0][j] = j * self.gap_penalty

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                match = dp[i - 1][j - 1] + self._pair_score(target_midi[i - 1], played_m[j - 1])
                delete = dp[i - 1][j] + self.gap_penalty
                insert = dp[i][j - 1] + self.gap_penalty
                dp[i][j] = max(match, delete, insert)

        # Backtracking to reconstruct alignment
        alignment = []
        i, j = N, M
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                score = self._pair_score(target_midi[i - 1], played_m[j - 1])
                if dp[i][j] == dp[i - 1][j - 1] + score:
                    target_note = target_notes[i - 1]
                    played_info = played_notes[j - 1]
                    played_note = played_info["note"]
                    correct = target_midi[i - 1] == played_m[j - 1]
                    alignment.append({
                        "type": "match" if correct else "substitution",
                        "target_index": i - 1,
                        "target_note": target_note,
                        "played_note": played_note,
                        "played_info": played_info,
                        "octave_error": (not correct) and same_pitch_class(target_midi[i - 1], played_m[j - 1]),
                        "feedback": "Correct!" if correct else f"Played {played_note} (expected {target_note})",
                    })
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + self.gap_penalty):
                alignment.append({
                    "type": "deletion",
                    "target_index": i - 1,
                    "target_note": target_notes[i - 1],
                    "played_note": None,
                    "played_info": None,
                    "octave_error": False,
                    "feedback": f"Missed {target_notes[i - 1]}",
                })
                i -= 1
            else:
                alignment.append({
                    "type": "insertion",
                    "target_index": None,
                    "target_note": None,
                    "played_note": played_notes[j - 1]["note"],
                    "played_info": played_notes[j - 1],
                    "octave_error": False,
                    "feedback": f"Extra note {played_notes[j - 1]['note']} played",
                })
                j -= 1

        alignment.reverse()

        beats = list(target_durations) if target_durations else [1.0] * N
        scoring = self._score_alignment(alignment, target_notes, beats, tempo_bpm, drill_type)
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

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_alignment(
        self,
        alignment: List[Dict[str, Any]],
        target_notes: List[str],
        beats: List[float],
        tempo_bpm: Optional[float],
        drill_type: Optional[str],
    ) -> Dict[str, Any]:
        N = len(target_notes)
        correct_count = sum(1 for step in alignment if step["type"] == "match")
        octave_count = sum(1 for step in alignment if step["octave_error"])
        exact_score = (correct_count / N) if N else 0.0
        # Credit for the right note name, regardless of octave.
        pitch_class_score = ((correct_count + octave_count) / N) if N else 0.0

        if drill_type == "single_note" and N == 1:
            pitch_score = self._score_single_note(alignment)
            return {
                "accuracy": pitch_score,
                "pitch_score": pitch_score,
                "timing_feedback": "Rhythm is not scored for single-note drills.",
            }

        result: Dict[str, Any] = {}
        if drill_type == "interval" and N >= 2:
            interval_score = self._score_interval(alignment)
            pitch_score = 0.7 * exact_score + 0.3 * pitch_class_score
            accuracy = 0.6 * pitch_score + 0.4 * interval_score
            result["interval_score"] = interval_score
            timing_feedback = "Focus on pitch for now — rhythm is secondary for intervals."
        elif drill_type in ("single_note", "interval"):
            pitch_score = exact_score
            accuracy = pitch_score
            timing_feedback = (
                "Rhythm is not scored for this drill type."
                if drill_type == "interval"
                else "Rhythm is not scored for single-note drills."
            )
        else:
            pitch_score = exact_score
            rhythm = self._rhythm_analysis(alignment, beats, tempo_bpm)
            weight = 0.15 if drill_type == "motif" else 0.3
            accuracy = (1 - weight) * pitch_score + weight * rhythm["score"]
            result["timing_score"] = rhythm["score"]
            timing_feedback = rhythm["feedback"]

        result.update({
            "accuracy": float(accuracy),
            "pitch_score": float(pitch_score),
            "timing_feedback": timing_feedback,
        })
        return result

    def _score_single_note(self, alignment: List[Dict[str, Any]]) -> float:
        for step in alignment:
            if step["type"] == "match":
                return 1.0
        for step in alignment:
            if step["octave_error"]:
                return 0.5
        return 0.0

    def _score_interval(self, alignment: List[Dict[str, Any]]) -> float:
        """1.0 if the played distance between the first two target notes is right.

        Uses the notes aligned to target positions 0 and 1, so a stray extra
        note at the start doesn't throw off the comparison.
        """
        by_target = {
            step["target_index"]: step
            for step in alignment
            if step["type"] in ("match", "substitution")
        }
        if 0 not in by_target or 1 not in by_target:
            return 0.0
        first, second = by_target[0], by_target[1]
        target_semitones = note_to_midi(second["target_note"]) - note_to_midi(first["target_note"])
        played_semitones = played_midi(second["played_info"]) - played_midi(first["played_info"])
        return 1.0 if target_semitones == played_semitones else 0.0

    def _rhythm_analysis(
        self,
        alignment: List[Dict[str, Any]],
        beats: List[float],
        tempo_bpm: Optional[float],
    ) -> Dict[str, Any]:
        """Compare played note spacing with the written rhythm.

        For each pair of consecutive correctly-played notes, the time between
        their starts divided by the written beats between them gives an
        estimate of seconds-per-beat. Steady, correct rhythm makes those
        estimates agree (regardless of overall tempo); their median compared
        with the drill's tempo tells us if the player is rushing or dragging.
        """
        matched = [
            (step["target_index"], step["played_info"]["start_time"])
            for step in alignment
            if step["type"] == "match" and step["played_info"] is not None
        ]
        spb = []
        for (ti, t_start), (tj, t_next) in zip(matched, matched[1:]):
            written = sum(beats[ti:tj])
            if written > 0 and t_next > t_start:
                spb.append((t_next - t_start) / written)

        if len(spb) < 2:
            return {
                "score": 0.5,
                "feedback": "Unable to evaluate timing. Try playing with a steadier tempo.",
            }

        spb = np.array(spb)
        coef_var = float(np.std(spb) / np.mean(spb))
        score = float(max(0.0, 1.0 - coef_var))

        speed_desc = "steady"
        if tempo_bpm:
            ratio = float(np.median(spb)) / (60.0 / tempo_bpm)
            if ratio < self.RUSHING_FACTOR:
                speed_desc = "rushing (faster than the reference tempo)."
            elif ratio > self.DRAGGING_FACTOR:
                speed_desc = "dragging (slower than the reference tempo)."

        if coef_var > 0.4:
            feedback = (
                f"Your rhythm is a bit uneven compared with the reference "
                f"(variation: {coef_var:.2f}). Try practicing with a metronome."
            )
        elif speed_desc == "steady":
            feedback = "Your timing and rhythm are great! Nice, steady play."
        else:
            feedback = f"Your rhythm is stable, but you are {speed_desc} Try adjusting your overall pace."
        return {"score": score, "feedback": feedback}

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
                if step["octave_error"]:
                    tips.append(f"Right note, wrong octave: you played {played}, the target was {target}.")
                elif drill_type == "single_note":
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
