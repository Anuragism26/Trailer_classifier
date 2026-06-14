import os
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_classifier():
    csv_path = os.path.join("data", "dataset_features.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    X = df[["cuts_per_minute", "mean_loudness", "spikes_per_minute"]]
    y = df["label"]
    
    print(f"📊 Dataset Loaded: {len(df)} trailers.")
    
    # Use simpler parameters since our dataset is small (prevents overfitting)
    model = RandomForestClassifier(n_estimators=15, max_depth=3, random_state=42)
    
    # Set up 4-Fold Cross Validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    print("🤖 Evaluating model using 4-Fold Cross-Validation...")
    scores = cross_val_score(model, X, y, cv=kf)
    
    print("\n🎯 --- MODEL PERFORMANCE REPORT ---")
    print(f"Cross-Validation Accuracies: {[round(s*100, 1) for s in scores]}%")
    print(f"Average System Accuracy: {scores.mean() * 100:.1f}%")
    
    # Train on ALL 13 trailers for the final production model save
    model.fit(X, y)
    
    model_output_path = os.path.join("src", "trailer_classifier_model.pkl")
    joblib.dump(model, model_output_path)
    print(f"\n💾 Final production model saved to {model_output_path}!")

if __name__ == "__main__":
    train_classifier()