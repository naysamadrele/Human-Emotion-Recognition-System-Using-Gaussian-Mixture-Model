import os
import librosa
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_path = "C:/Users/Naysa/Documents/RAVDESS_Emotional_Speech"

emotion_labels = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

def extract_features(file_path, n_mfcc=13):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

X, y = [], []
for actor_folder in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor_folder)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file)
            print(f"Processing: {file_path}")
            try:
                emotion_code = file.split("-")[2]
                emotion = emotion_labels.get(emotion_code)
                if emotion:
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
            except IndexError:
                print(f"Skipping file due to format issue: {file}")
                continue

X = np.array(X)
y = np.array(y)
print(f"✅ Extracted {len(X)} feature vectors.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gmm_models = {}
for emotion in np.unique(y_train):
    gmm = GaussianMixture(n_components=4, max_iter=200, covariance_type='diag', random_state=42)
    emotion_data = X_train[y_train == emotion]
    gmm.fit(emotion_data)
    gmm_models[emotion] = gmm

print("✅ Model Training Complete!")

def calculate_accuracy():
    predictions = []
    for features in X_test:
        scores = {emotion: gmm.score(features.reshape(1, -1)) for emotion, gmm in gmm_models.items()}
        predicted_emotion = max(scores, key=scores.get)
        predictions.append(predicted_emotion)
    
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"🎯 Model Accuracy: {accuracy:.2f}%")
    return accuracy

accuracy = calculate_accuracy()

model_path = "C:/Users/Naysa/Documents/gmm_models.pkl"
with open(model_path, "wb") as f:
    pickle.dump(gmm_models, f)
print("✅ GMM Models Saved Successfully!")

with open(model_path, "rb") as f:
    gmm_models = pickle.load(f)

def predict_emotion():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a .wav file", filetypes=[("WAV Files", "*.wav")])
    if not file_path:
        print("⚠ No file selected!")
        return
    
    features = extract_features(file_path).reshape(1, -1)
    scores = {emotion: gmm.score(features) for emotion, gmm in gmm_models.items()}
    predicted_emotion = max(scores, key=scores.get)
    print(f"🎭 Predicted Emotion: {predicted_emotion}")
    print(f"🎯 Model Accuracy: {accuracy:.2f}%")

if _name_ == "_main_":
    predict_emotion()
