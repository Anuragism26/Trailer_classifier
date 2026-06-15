import streamlit as st
import cv2
import joblib
import numpy as np
import os
import yt_dlp

st.set_page_config(page_title="Cinematic ML Insights", layout="centered")

st.title("Multimodal Data Collection & Feature Engine")
st.write("Extract cinematic patterns from files, log YouTube tracking metadata, and run Random Forest predictions.")

if "data_log" not in st.session_state:
    st.session_state.data_log = []

st.markdown("## Channel 1: Metadata Harvesting (YouTube URL)")
yt_url = st.text_input("Paste any YouTube link to fetch metadata instantly:")

collected_features = None
source_name = ""

if yt_url:
    with st.spinner("Fetching public tracking metadata..."):
        ydl_opts = {'extract_flat': True, 'quiet': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(yt_url, download=False)
                
                st.success("Metadata Successfully Collected!")
                source_name = info.get('title', 'Unknown Title')
                
                col1, col2 = st.columns(2)
                col1.metric("Video Title", source_name[:35] + "...")
                col2.metric("View Count", f"{info.get('view_count', 0):,}")
                
                st.markdown("---")
                st.info("Since video downloading is disabled to protect cloud resources, tweak the sliders below to estimate the cinematic profile and test the ML model!")
                
                sim_cuts = st.slider("Estimated Visual Cuts / Min", 0.0, 60.0, 25.0, 0.5)
                sim_loud = st.slider("Estimated Audio Loudness (RMS)", 0.001, 0.150, 0.035, 0.001, format="%.3f")
                sim_spikes = st.slider("Estimated Audio Jump Scares / Min", 0.0, 50.0, 15.0, 0.5)
                
                collected_features = [sim_cuts, sim_loud, sim_spikes]
                
        except Exception as e:
            st.error(f"Could not fetch metadata: {e}")

st.markdown("---")
st.markdown("## Channel 2: Live Feature Extraction (Local File)")
uploaded_file = st.file_uploader("Upload an MP4 file to parse frame pacing and waveforms directly:", type=["mp4"])

temp_path = "temp_uploaded_trailer.mp4"

if uploaded_file is not None and not yt_url:
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
        
    source_name = uploaded_file.name
    
    with st.spinner("Extracting math-driven audio/video matrices..."):
        from src.extract_features import get_video_features, get_audio_features
        try:
            cuts_pm, _ = get_video_features(temp_path)
            loudness, spikes_pm = get_audio_features(temp_path)
            collected_features = [cuts_pm, loudness, spikes_pm]
            
            st.success("Extracted features flawlessly from file matrices!")
        except Exception as e:
            st.error(f"Feature processing error: {e}")
            
    if os.path.exists(temp_path):
        os.remove(temp_path)

if collected_features:
    st.markdown("---")
    st.markdown("## Model Evaluation Dashboard")
    
    model_path = os.path.join("src", "trailer_classifier_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        cuts, loud, spikes = collected_features
        input_vector = np.array([[cuts, loud, spikes]])
        
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        
        st.success(f"Model Prediction: **{prediction.upper()}**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Visual Cuts / Min", f"{cuts:.1f}")
        c2.metric("RMS Loudness", f"{loud:.4f}")
        c3.metric("Spikes / Min", f"{spikes:.1f}")
        
        if st.button("Append Row to Feature Dataset Collection"):
            new_row = {
                "Source": source_name,
                "Cuts_Per_Min": round(cuts, 2),
                "RMS_Loudness": round(loud, 4),
                "Spikes_Per_Min": round(spikes, 2),
                "Predicted_Genre": prediction.upper()
            }
            st.session_state.data_log.append(new_row)
            st.balloons()
    else:
        st.error("Error: Production model target file missing at 'src/trailer_classifier_model.pkl'")

if st.session_state.data_log:
    st