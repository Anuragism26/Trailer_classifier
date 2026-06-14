import cv2
import numpy as np

def get_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    
    ret, prev_frame = cap.read()
    if not ret:
        return 0, 0
        
    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cut_count = 0
    
    # Threshold for a "cut" (you can tweak this after testing)
    CUT_THRESHOLD = 30.0 
    
    is_prev_dark = False  # Track if previous frame was pitch black
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check if the current frame is a dark/black screen transition
        avg_brightness = np.mean(gray)
        is_current_dark = avg_brightness < 12.0 
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        mean_diff = np.mean(frame_diff)
        
        # CASE 1: Standard sudden cut
        if mean_diff > CUT_THRESHOLD:
            cut_count += 1
            
        # CASE 2: Sudden cut coming OUT of a dark/black screen (Horror trailer style)
        elif is_prev_dark and not is_current_dark and mean_diff > 15.0:
            cut_count += 1
            
        # Update states for the next iteration loop
        prev_gray = gray
        is_prev_dark = is_current_dark
        
    cap.release()
    if mean_diff > 15.0:  # print anything that shows moderate change
        print(f"Frame difference detected: {mean_diff}")
    
    # Normalize features by time
    cuts_per_minute = (cut_count / duration_seconds) * 60
    return cuts_per_minute, duration_seconds

import librosa
from moviepy import VideoFileClip
import os

def get_audio_features(video_path):
    temp_wav = "temp_audio.wav"
    try:
        # 1. Use MoviePy to extract the audio track and write it to a clean WAV file
        video = VideoFileClip(video_path)
        if video.audio is None:
            print(f"No audio track found in {video_path}")
            return 0.0, 0.0
            
        # Write audio out quietly without terminal spam
        video.audio.write_audiofile(temp_wav, logger=None)
        video.close() # Free up the file system lock
        
        # 2. Load the clean WAV file into Librosa seamlessly
        y, sr = librosa.load(temp_wav, sr=None)
        
    except Exception as e:
        print(f"Error extracting/loading audio for {video_path}: {e}")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return 0.0, 0.0

    # 3. Calculate Root-Mean-Square (RMS) Energy for overall loudness
    rms = librosa.feature.rms(y=y)
    mean_loudness = np.mean(rms)
    
    # 4. Track sudden audio volume jumps (Onset Strength)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Pick out major peaks/spikes in the audio
    peaks = librosa.util.peak_pick(
        onset_env, 
        pre_max=7, post_max=7, 
        pre_avg=7, post_avg=7, 
        delta=4.0, wait=10
    )
    
    # Calculate duration to find spikes per minute
    duration_seconds = len(y) / sr
    spikes_per_minute = (len(peaks) / duration_seconds) * 60
    
    # 5. Clean up the temporary WAV file from your workspace
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        
    return float(mean_loudness), float(spikes_per_minute)



if __name__ == "__main__":
    import os
    import pandas as pd

    TRAILERS_FOLDER = os.path.join("data", "raw_trailers")
    data_rows = []

    print("🚀 Starting Multimodal Feature Extraction (Audio + Video)...")

    for filename in os.listdir(TRAILERS_FOLDER):
        if filename.endswith(".mp4"):
            video_path = os.path.join(TRAILERS_FOLDER, filename)
            print(f"\n🎬 Processing: {filename}")
            
            # Extract Video Features (Phase 2)
            cuts_pm, duration = get_video_features(video_path)
            
            # Extract Audio Features (Phase 3)
            loudness, spikes_pm = get_audio_features(video_path)
            
            print(f"   [Video] Pacing: {cuts_pm:.1f} cuts/min")
            print(f"   [Audio] Loudness: {loudness:.4f} | Spikes: {spikes_pm:.1f} per min")
            
            # Store data rows for the next phase (Machine Learning)
            data_rows.append({
                "filename": filename,
                "cuts_per_minute": cuts_pm,
                "mean_loudness": loudness,
                "spikes_per_minute": spikes_pm
            })

    # Save to a temporary CSV check
    df = pd.DataFrame(data_rows)
    df.to_csv(os.path.join("data", "dataset_features.csv"), index=False)
    print("\n✅ All features extracted and saved to data/dataset_features.csv!")