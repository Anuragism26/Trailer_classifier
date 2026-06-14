import streamlit as st
import cv2
import joblib
import numpy as np
import os
import yt_dlp  # Clean, native Python library to stream from YouTube

# Import your custom feature extractors from Phase 2 & 3
from src.extract_features import get_video_features, get_audio_features

# Page Configuration
st.set_page_config(page_title="Cinematic ML Classifier", page_icon="🎬", layout="centered")

st.title("🎬 Multimodal Film Trailer Genre Classifier")
st.write("Provide a movie trailer to let AI analyze its editing pacing and audio dynamics live.")

# --- THE UPDATED INPUT SECTION ---
yt_url = st.text_input("🔗 Option 1: Paste a YouTube Trailer URL:")
uploaded_file = st.file_uploader("📂 Option 2: Or choose a local MP4 file...", type=["mp4"])

temp_path = "temp_uploaded_trailer.mp4"

# Step 1: Handle YouTube Link Input
if yt_url:
    with st.spinner("📥 Streaming video layout directly from YouTube..."):
        # We tell yt_dlp to look for the worst video quality (fast download) but best audio track
        # Force yt-dlp to grab a pre-merged standalone mp4 file so it doesn't need FFmpeg
        opts = {
            'format': 'best[ext=mp4]/mp4', 
            'outtmpl': temp_path, 
            'quiet': True
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([yt_url])
        except Exception as e:
            st.error(f"❌ Failed to parse YouTube link: {e}")
            yt_url = None

# Step 2: Handle Local File Drop Box Input (If no YouTube link is present)
elif uploaded_file is not None:
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())


# --- THE MAIN PROCESSING AND ML BLOCK ---
# This executes flawlessly regardless of whether the video came from Option 1 or Option 2
if (yt_url and os.path.exists(temp_path)) or uploaded_file is not None:
    # Render the video player in your web browser interface
    st.video(temp_path)
    st.markdown("---")
    
    with st.spinner("🤖 Extracting multimodal patterns... Please wait."):
        # Run your Phase 2 & 3 extractors on the saved video path
        cuts_pm, _ = get_video_features(temp_path)
        loudness, spikes_pm = get_audio_features(temp_path)
        
        # Load your trained model file
        model_path = os.path.join("src", "trailer_classifier_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Format extracted features into a 2D numpy array for the model
            features = np.array([[cuts_pm, loudness, spikes_pm]])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_
            
            # Display Final Prediction Badge
            st.success(f"### 🎯 Predicted Genre: **{prediction.upper()}**")
            
            # Display Metrics Dashboard
            st.subheader("📊 Extracted Cinematic Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Visual Cuts / Min", f"{cuts_pm:.1f}")
            col2.metric("Avg Sound Volume (RMS)", f"{loudness:.4f}")
            col3.metric("Audio Spikes / Min", f"{spikes_pm:.1f}")
            
            # Display Prediction Confidence Bars
            st.write("")
            st.subheader("🔮 Model Confidence Breakdown")
            for cls, prob in zip(classes, probabilities):
                st.write(f"**{cls.capitalize()}**")
                st.progress(float(prob))
        else:
            st.error("Error: Could not locate 'src/trailer_classifier_model.pkl'. Run your training script first!")

    # Clean up the temporary file from your system workspace storage
    if os.path.exists(temp_path):
        os.remove(temp_path)