import streamlit as st
import numpy as np
import io
import os
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

# Import local modules
from src.database.drills import (
    DRILL_LEVELS,
    get_drills_by_level,
    duration_to_seconds,
)
from src.audio_processing.librosa_pyin import LibrosaPyinTracker
from src.evaluation.melody_aligner import MelodyAligner

# Ensure temp directory exists for file handling
os.makedirs("temp", exist_ok=True)

# -----------------
# Premium Design CSS
# -----------------
st.set_page_config(
    page_title="Play-by-Ear Tutor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-title {
        font-size: 3rem;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #8A8D93;
        margin-bottom: 2rem;
    }
    
    /* Card Glassmorphism style */
    .dashboard-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    .accuracy-ring {
        display: flex;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle, rgba(77,150,255,0.1) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        width: 150px;
        height: 150px;
        border: 6px solid #4D96FF;
        margin: 0 auto;
        font-size: 2rem;
        font-weight: 800;
        color: #4D96FF;
    }
    
    .feedback-correct {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .feedback-incorrect {
        color: #FF5252;
        font-weight: bold;
    }
    
    .feedback-extra {
        color: #FFC107;
        font-weight: bold;
    }
    
    .feedback-missed {
        color: #9E9E9E;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# Musical Helpers
# -----------------
NOTE_FREQS = {
    'C4': 261.63, 'C#4': 277.18, 'Db4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'Eb4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'Gb4': 369.99, 'G4': 392.00, 'G#4': 415.30,
    'Ab4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'Bb4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'Db5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'Eb5': 622.25,
    'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 'Gb5': 739.99, 'G5': 783.99, 'G#5': 830.61,
    'Ab5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'Bb5': 932.33, 'B5': 987.77,
}

FINGERING_DATABASE = {
    'C4': {'guitar': 'String 5, Fret 3', 'piano': 'Middle C (first white key left of 2 black keys)'},
    'D4': {'guitar': 'String 4, Open', 'piano': 'White key between the 2 black keys'},
    'E4': {'guitar': 'String 4, Fret 2', 'piano': 'White key right of the 2 black keys'},
    'F4': {'guitar': 'String 4, Fret 3', 'piano': 'White key left of 3 black keys'},
    'G4': {'guitar': 'String 3, Open', 'piano': 'First white key inside 3 black keys'},
    'A4': {'guitar': 'String 3, Fret 2', 'piano': 'Second white key inside 3 black keys'},
    'B4': {'guitar': 'String 2, Open', 'piano': 'White key right of 3 black keys'},
    'C5': {'guitar': 'String 2, Fret 1', 'piano': 'First white key left of 2 black keys (octave up)'},
}

def get_fingering(note: str) -> dict:
    return FINGERING_DATABASE.get(note, {'guitar': 'Check chord diagram', 'piano': 'Varies'})

def generate_reference_audio(drill: dict, sample_rate=22050) -> bytes:
    """Synthesize reference drill into a clean sine wave audio track."""
    tempo_bpm = drill.get("tempo_bpm", 60)
    audio_data = []

    for note, dur in zip(drill["notes"], drill["durations"]):
        freq = NOTE_FREQS.get(note, 261.63)
        note_seconds = duration_to_seconds(dur, tempo_bpm)
        t = np.linspace(0, note_seconds, int(sample_rate * note_seconds), endpoint=False)
        fade = np.linspace(1.0, 0.01, len(t))
        wave = np.sin(2 * np.pi * freq * t) * fade
        audio_data.extend(wave)

        silence = np.zeros(int(sample_rate * 0.05))
        audio_data.extend(silence)

    audio_array = np.array(audio_data, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='wav')
    return buffer.getvalue()


DRILL_TYPE_LABELS = {
    "single_note": "Single Note",
    "interval": "Interval",
    "motif": "Short Motif",
    "melody": "Melody",
}

# -----------------
# Sidebar Controls
# -----------------
st.sidebar.markdown("### 🛠️ Configuration")
pitch_tracker_type = st.sidebar.selectbox("Pitch Tracker Backend", ["Librosa pYIN", "Spotify Basic Pitch (Future)"])
alignment_strictness = st.sidebar.slider("Alignment Match Bonus", 1, 5, 2)

st.sidebar.markdown("""
---
### 💡 How to Play
1. **Choose a level** and drill from the Ear Training curriculum.
2. **Listen** to the reference track.
3. **Record or Upload** your attempt.
4. **Review** pitch, interval, and rhythm feedback tailored to the drill type.
""")

# -----------------
# Main Interface
# -----------------
st.markdown("<h1 class='main-title'>🎵 Play-by-Ear Tutor</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ear Training Fundamentals — hear a note or phrase, reproduce it, and get instant feedback.</div>", unsafe_allow_html=True)

# 1. Drill Selection Card
st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
st.subheader("Step 1: Choose Your Drill")

filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    selected_level = st.selectbox(
        "Training level:",
        options=sorted(DRILL_LEVELS.keys()),
        format_func=lambda lvl: f"Level {lvl} — {DRILL_LEVELS[lvl]}",
    )
with filter_col2:
    level_drills = get_drills_by_level(selected_level)
    available_types = sorted({d["type"] for d in level_drills})
    selected_type = st.selectbox(
        "Drill type:",
        options=["all"] + available_types,
        format_func=lambda t: "All types" if t == "all" else DRILL_TYPE_LABELS.get(t, t),
    )

filtered_drills = [
    d for d in level_drills
    if selected_type == "all" or d["type"] == selected_type
]

selected_drill_name = st.selectbox(
    "Select a drill:",
    options=[d["name"] for d in filtered_drills],
)
selected_drill = next(d for d in filtered_drills if d["name"] == selected_drill_name)

col1, col2 = st.columns([2, 1])
with col1:
    drill_type_label = DRILL_TYPE_LABELS.get(selected_drill["type"], selected_drill["type"])
    st.write(f"**Level:** {selected_drill['level']} — {DRILL_LEVELS[selected_drill['level']]}")
    st.write(f"**Type:** {drill_type_label}")
    st.write(f"**Difficulty:** {selected_drill['difficulty']}")
    st.write(f"**Description:** {selected_drill['description']}")
    if selected_drill.get("interval_name"):
        st.write(f"**Interval:** {selected_drill['interval_name']}")
    st.write(f"**Target Notes:** `{' ➔ '.join(selected_drill['notes'])}`")
with col2:
    ref_audio_bytes = generate_reference_audio(selected_drill)
    st.write("🔈 Reference Guide:")
    st.audio(ref_audio_bytes, format="audio/wav")
st.markdown("</div>", unsafe_allow_html=True)

# 2. Recording/Upload Card
st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
st.subheader("Step 2: Record / Upload Your Performance")
st.info("Listen to the reference, then play or sing the target on your instrument. Keep background noise minimal.")

record_label = {
    "single_note": "Record your note via microphone",
    "interval": "Record both notes via microphone",
    "motif": "Record the motif via microphone",
    "melody": "Record your performance via microphone",
}.get(selected_drill["type"], "Record your performance via microphone")

# Support Streamlit standard audio input (mic recorder)
recorded_file = None
if hasattr(st, "audio_input"):
    recorded_file = st.audio_input(record_label)

# Fallback: File Uploader
uploaded_file = st.file_uploader(
    "Alternatively, upload a WAV or MP3 recording:",
    type=["wav", "mp3", "m4a"]
)

input_file = recorded_file if recorded_file is not None else uploaded_file

st.markdown("</div>", unsafe_allow_html=True)

# 3. Analyze & Show Results
if input_file is not None:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.subheader("Step 3: AI Musical Feedback")
    
    with st.spinner("Processing audio & tracking pitch..."):
        # Save temp file
        temp_path = os.path.join("temp", "input_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(input_file.read())
            
        try:
            # Transcribe
            tracker = LibrosaPyinTracker()
            detected_notes = tracker.transcribe(temp_path)
            
            # Align
            aligner = MelodyAligner(match_score=alignment_strictness)
            results = aligner.align(
                selected_drill["notes"],
                detected_notes,
                drill_type=selected_drill["type"],
            )
            
            # Display score & results
            score_col, detail_col = st.columns([1, 2])
            
            with score_col:
                acc_percentage = int(results["accuracy"] * 100)
                st.markdown(f"<div class='accuracy-ring'>{acc_percentage}%</div>", unsafe_allow_html=True)
                st.write("")
                st.markdown(f"<p style='text-align:center;'><b>Accuracy Score</b></p>", unsafe_allow_html=True)
                
            with detail_col:
                st.markdown("### Performance Overview")
                st.write(f"🏁 **Pitch Tracking Status:** Success")
                st.write(f"🎯 **Pitch Score:** {int(results['pitch_score'] * 100)}%")
                if results.get("interval_score") is not None:
                    st.write(f"🎼 **Interval Score:** {int(results['interval_score'] * 100)}%")
                if results.get("timing_score") is not None:
                    st.write(f"⏱️ **Rhythm Score:** {int(results['timing_score'] * 100)}%")
                st.write(f"⏱️ **Rhythm & Timing:** {results['timing_feedback']}")
                
                st.markdown("#### Actionable Practice Tips:")
                for tip in results["tips"]:
                    st.write(f"- 💡 {tip}")
                    
            st.markdown("---")
            st.markdown("### Note-by-Note Breakdown")
            
            # Build alignment table layout
            table_cols = st.columns(len(results["alignment"]))
            for idx, step in enumerate(results["alignment"]):
                with table_cols[idx]:
                    if step["type"] == "match":
                        st.markdown(f"<div style='text-align:center; padding: 10px; border-radius: 8px; background-color: rgba(76,175,80,0.15);'><span class='feedback-correct'>{step['target_note']}</span><br><small>Correct</small></div>", unsafe_allow_html=True)
                    elif step["type"] == "substitution":
                        st.markdown(f"<div style='text-align:center; padding: 10px; border-radius: 8px; background-color: rgba(255,82,82,0.15);'><span class='feedback-incorrect'>{step['played_note']}</span><br><small>Expected {step['target_note']}</small></div>", unsafe_allow_html=True)
                    elif step["type"] == "deletion":
                        st.markdown(f"<div style='text-align:center; padding: 10px; border-radius: 8px; background-color: rgba(158,158,158,0.15);'><span class='feedback-missed'>-</span><br><small>Missed {step['target_note']}</small></div>", unsafe_allow_html=True)
                    elif step["type"] == "insertion":
                        st.markdown(f"<div style='text-align:center; padding: 10px; border-radius: 8px; background-color: rgba(255,193,7,0.15);'><span class='feedback-extra'>{step['played_note']}</span><br><small>Extra Note</small></div>", unsafe_allow_html=True)
                        
            # Pitch Tracker Visualizations
            st.markdown("---")
            st.markdown("### Pitch Tracking Graph")
            
            # Load audio for plotting
            y, sr = librosa.load(temp_path, sr=22050)
            times = librosa.times_like(y, sr=sr)
            
            fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='none')
            ax.set_facecolor('#1E1E1E')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            # Plot reference pitches as horizontal markers/lines
            ref_notes = selected_drill["notes"]
            ref_hz_values = [NOTE_FREQS.get(n, 261.63) for n in ref_notes]
            for note_name, hz in zip(ref_notes, ref_hz_values):
                ax.axhline(hz, color='gray', linestyle='--', alpha=0.5)
                ax.text(0.1, hz + 5, note_name, color='gray', fontsize=8)
                
            # Plot transcribed pitches
            if detected_notes:
                p_times = []
                p_freqs = []
                for note_dict in detected_notes:
                    p_times.extend([note_dict["start_time"], note_dict["end_time"]])
                    p_freqs.extend([note_dict["frequency"], note_dict["frequency"]])
                ax.plot(p_times, p_freqs, 'o-', color='#4D96FF', label='Played Pitch', linewidth=2)
                
            ax.set_title("Played Frequency (Hz) vs Reference Notes", color='white', fontsize=12)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Frequency (Hz)")
            ax.legend(facecolor='#1E1E1E', labelcolor='white')
            st.pyplot(fig)
            
            # Fingering Help Panel
            st.markdown("---")
            st.markdown("### 🎸 Suggested Instrument Fingerings")
            st.write("Struggling with some notes? Here's how to play the reference notes on different instruments:")
            
            fingering_cols = st.columns(4)
            for idx, note in enumerate(ref_notes[:8]): # limit columns to first 8 notes for presentation
                with fingering_cols[idx % 4]:
                    f_info = get_fingering(note)
                    st.markdown(f"""
                    **Note {note}**
                    - *Guitar:* `{f_info['guitar']}`
                    - *Piano:* `{f_info['piano']}`
                    """)
                    
        except Exception as e:
            st.error(f"Error processing audio. Please make sure the audio file is clear and formatted correctly. Details: {e}")
            
    st.markdown("</div>", unsafe_allow_html=True)
