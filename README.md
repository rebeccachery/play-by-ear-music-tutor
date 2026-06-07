# Play-by-Ear Music Instrument Tutor
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Type](https://img.shields.io/badge/type-play--by--ear%20tutor-teal)
![Input](https://img.shields.io/badge/input-live%20or%20uploaded%20audio-red)
![Pitch](https://img.shields.io/badge/pitch-Librosa%20pYIN-blue)
![Feedback](https://img.shields.io/badge/feedback-pitch%20%2B%20rhythm-green)
![Feature](https://img.shields.io/badge/feature-note--by--note%20analysis-purple)
![Feature](https://img.shields.io/badge/feature-fingering%20help-orange)
![Frontend](https://img.shields.io/badge/frontend-Streamlit-FF4B4B)

This is a Streamlit-based web application that serves as a play-by-ear music instrument tutor. It listens to your playing, transcribes the notes using Librosa's pYIN pitch tracker, aligns them to a reference melody using sequence alignment, and provides detailed feedback on both pitch accuracy and rhythm.

## Features

- **Reference Player**: Listen to the reference exercise melody generated dynamically via synthesized sine wave tones.
- **Microphone / File Input**: Record directly in the browser or upload your `.wav` / `.mp3` / `.m4a` file.
- **Visual Pitch Comparison**: Check your note path plotted against the expected notes on an interactive frequency chart.
- **Rhythm Feedback**: Detects if your rhythm is rushing, dragging, or steady.
- **Note-by-Note Breakdown**: Clear highlights indicating correct notes, missed notes, incorrect substitutions, and extra notes.
- **Fingering Help Panel**: Quick piano/guitar tabs reference to practice notes you struggle with.

## Setup and Installation

1. Install the Python dependencies (make sure you use the `--only-binary` flag for `llvmlite` and `numba` to prevent compile errors on macOS):
   ```bash
   pip install --only-binary :all: llvmlite numba
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open the local address in your web browser (typically `http://localhost:8501`).
