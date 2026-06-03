from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePitchTracker(ABC):
    """
    Abstract base class for audio transcription / pitch tracking engines.
    Allows easy swap-in of different backends (Librosa, CREPE, Basic Pitch, etc.)
    """
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe the audio file at audio_path into a list of notes.
        
        Args:
            audio_path (str): Path to the target audio file.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the detected notes.
                Each dictionary must have:
                - 'note': str (e.g. 'C4')
                - 'frequency': float (pitch in Hz)
                - 'start_time': float (seconds)
                - 'end_time': float (seconds)
        """
        pass
