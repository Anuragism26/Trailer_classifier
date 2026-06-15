
# Multimodal Film Trailer Genre Classifier

A machine learning pipeline and interactive web dashboard that classifies movie trailers into genres (**Action**, **Horror**, or **Drama**) by analyzing live cinematic visual and audio footprints. 
Instead of relying on heavy deep learning, this project extracts precise technical features such as cuts per min, frame rate and audio spikefrom media files to train a highly lightweight and explainable **Random Forest Classifier**.

[Launch the Web App](https://trailer-genre-classifier.streamlit.app)
##  Project Architecture
```text
├── data/
│   └── dataset_features.csv          # Compiled dataset with extracted features & labels
├── src/
│   ├── extract_features.py           # OpenCV & Librosa audio-video extraction engine
│   ├── train_model.py                # Random Forest training & Cross-Validation pipeline
│   └── trailer_classifier_model.pkl  # Pre-trained production ML model
├── app.py                            # Streamlit Web UI Dashboard (YouTube + File upload)
└── requirements.txt                  # Python dependencies
