from typing import List, Dict, Any, Tuple
import numpy as np

class MelodyAligner:
    """
    Aligns the played note sequence with the expected/target note sequence
    and generates detailed musical feedback on pitch accuracy and timing.
    """
    def __init__(self, match_score: int = 2, mismatch_score: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_score = mismatch_score
        self.gap_penalty = gap_penalty

    def align(self, target_notes: List[str], played_notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Align target_notes (e.g., ['C4', 'E4', 'G4']) with played_notes dictionaries.
        Returns alignment score, detailed alignment path, and feedback.
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
        
        # Calculate pitch accuracy
        correct_count = sum(1 for step in alignment if step["type"] == "match")
        accuracy = (correct_count / N) if N > 0 else 0.0
        
        # Timing feedback
        timing_feedback = self._evaluate_timing(alignment)
        
        # Generate actionable advice
        tips = []
        for step in alignment:
            if step["type"] == "substitution":
                tips.append(f"Work on the transition/interval to {step['target_note']} (you played {step['played_note']} instead).")
            elif step["type"] == "deletion":
                tips.append(f"Make sure to play {step['target_note']}; it was missed.")
                
        if not tips:
            tips.append("Excellent work! You played all notes correctly and in the right order.")
            
        return {
            "alignment": alignment,
            "accuracy": accuracy,
            "timing_feedback": timing_feedback,
            "tips": list(set(tips))[:3] # Max 3 distinct tips
        }
        
    def _evaluate_timing(self, alignment: List[Dict[str, Any]]) -> str:
        """
        Evaluate user rhythm and speed based on matched notes.
        """
        matched_durations = []
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
