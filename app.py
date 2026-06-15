import os
import cv2
import joblib
import numpy as np
import streamlit as st
import yt_dlp

from src.extract_features import get_audio_features, get_video_features

# Page Configuration
st.set_page_config(page_title="Cinematic ML Classifier", layout="centered")

st.title("Multimodal Film Trailer Genre Classifier")
st.write("Provide a movie trailer link or file to analyze editing pacing and audio dynamics.")

# Input Section
yt_url = st.text_input("Option 1: Paste a YouTube Trailer URL:")
uploaded_file = st.file_uploader("Option 2: Or choose a local MP4 file...", type=["mp4"])

temp_path = "temp_uploaded_trailer.mp4"

# Handle YouTube Link Input
if yt_url:
    with st.spinner("Downloading video from YouTube..."):
        opts = {
            'format': 'best[ext=mp4]/mp4', 
            'outtmpl': temp_path, 
            'quiet': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['default', '-android_sdkless']
                }
            }
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([yt_url])
        except Exception as e:
            st.error(f"Failed to parse YouTube link: {e}")
            yt_url = None

# Handle Local File Input
elif uploaded_file is not None:
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

# Processing and Inference Block
if (yt_url and os.path.exists(temp_path)) or uploaded_file is not None:
    st.video(temp_path)
    st.markdown("---")
    
    with st.spinner("Extracting features and generating predictions..."):
        cuts_pm, _ = get_video_features(temp_path)
        loudness, spikes_pm = get_audio_features(temp_path)
        
        model_path = os.path.join("src", "trailer_classifier_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            features = np.array([[cuts_pm, loudness, spikes_pm]])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_
            
            # Display Results
            st.success(f"Predicted Genre: **{prediction.upper()}**")
            
            st.subheader("Extracted Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Visual Cuts / Min", f"{cuts_pm:.1f}")
            col2.metric("Avg Sound Volume (RMS)", f"{loudness:.4f}")
            col3.metric("Audio Spikes / Min", f"{spikes_pm:.1f}")
            
            st.write("")
            st.subheader("Model Confidence Breakdown")
            for cls, prob in zip(classes, probabilities):
                st.write(f"**{cls.capitalize()}**")
                st.progress(float(prob))
        else:
            st.error("Error: Could not locate 'src/trailer_classifier_model.pkl'. Verify that the model training pipeline has run successfully.")

    # Cleanup workspace
    if os.path.exists(temp_path):
        os.remove(temp_path)