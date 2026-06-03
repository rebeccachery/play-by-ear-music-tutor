import librosa
import numpy as np
from typing import List, Dict, Any
from .base import BasePitchTracker

class LibrosaPyinTracker(BasePitchTracker):
    """
    Pitch tracker implementation using Librosa's pYIN algorithm.
    Suitable for monophonic audio signals (singing, solo instrument plays).
    """
    def __init__(self, fmin: str = "C2", fmax: str = "C7", min_note_duration: float = 0.15):
        self.fmin_hz = librosa.note_to_hz(fmin)
        self.fmax_hz = librosa.note_to_hz(fmax)
        self.min_note_duration = min_note_duration

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        # Load audio (downsample to 22050 Hz for standard speech/music analysis)
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Calculate pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin_hz,
            fmax=self.fmax_hz,
            sr=sr,
            fill_na=None
        )
        
        # Get frame times
        hop_length = 512
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
        
        # Segment into discrete notes
        notes_segments = []
        current_note = None
        current_freqs = []
        start_time = None
        
        frame_duration = hop_length / sr
        
        for i, freq in enumerate(f0):
            # Check if voiced frame
            if freq is not None and not np.isnan(freq):
                note_name = librosa.hz_to_note(freq)
                
                if current_note is None:
                    # Start a new note segment
                    current_note = note_name
                    start_time = times[i]
                    current_freqs = [freq]
                elif note_name == current_note:
                    # Continue existing note segment
                    current_freqs.append(freq)
                else:
                    # Note changed - save previous note if it meets duration threshold
                    end_time = times[i]
                    duration = end_time - start_time
                    if duration >= self.min_note_duration:
                        notes_segments.append({
                            "note": current_note,
                            "frequency": float(np.median(current_freqs)),
                            "start_time": float(start_time),
                            "end_time": float(end_time)
                        })
                    
                    # Start new segment
                    current_note = note_name
                    start_time = times[i]
                    current_freqs = [freq]
            else:
                # Silence / unvoiced frame - save current note if active
                if current_note is not None:
                    end_time = times[i]
                    duration = end_time - start_time
                    if duration >= self.min_note_duration:
                        notes_segments.append({
                            "note": current_note,
                            "frequency": float(np.median(current_freqs)),
                            "start_time": float(start_time),
                            "end_time": float(end_time)
                        })
                    current_note = None
                    current_freqs = []
                    start_time = None
                    
        # Catch any final remaining note
        if current_note is not None:
            end_time = times[-1] + frame_duration
            duration = end_time - start_time
            if duration >= self.min_note_duration:
                notes_segments.append({
                    "note": current_note,
                    "frequency": float(np.median(current_freqs)),
                    "start_time": float(start_time),
                    "end_time": float(end_time)
                })
                
        return notes_segments
