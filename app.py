import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import librosa

# Define label columns globally for Task 2
label_columns = ['type_blocker', 'type_regression', 'type_bug', 'type_documentation', 
                 'type_enhancement', 'type_task', 'type_dependency_upgrade']

# Load models and preprocessing objects for Task 2
lr_multi = joblib.load('lr_multi_defect.pkl')
svm_multi = joblib.load('svm_multi_defect.pkl')
perceptron_multi = joblib.load('perceptron_multi_defect.pkl')
dnn_multi = tf.keras.models.load_model('dnn_multi_defect.h5')
scaler_defect = joblib.load('scaler_defect.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load models and preprocessing objects for Task 1
svm_deepfake = joblib.load('svm_deepfake.pkl')
lr_deepfake = joblib.load('lr_deepfake.pkl')
perceptron_deepfake = joblib.load('perceptron_deepfake.pkl')
dnn_deepfake = tf.keras.models.load_model('dnn_deepfake.h5')
scaler_audio = joblib.load('scaler_audio.pkl')

# Function to extract MFCC features for Task 1
def extract_mfcc(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    if mfcc.shape[1] > 300:
        mfcc = mfcc[:, :300]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])), mode='constant')
    return mfcc.flatten()

# Streamlit UI
st.title("Multi-Task Prediction App")

# Task 2: Defect Prediction
st.header("Defect Prediction")
text_input = st.text_area("Enter defect report text", height=100)
model_choice_defect = st.selectbox("Select Model for Defect Prediction", ["Logistic Regression", "SVM", "Perceptron", "DNN"])

if st.button("Predict Defect"):
    if text_input:
        # Transform input text using TF-IDF
        X = vectorizer.transform([text_input]).toarray()
        X_scaled = scaler_defect.transform(X)

        if model_choice_defect == "Logistic Regression":
            pred = lr_multi.predict(X_scaled)
        elif model_choice_defect == "SVM":
            pred = svm_multi.predict(X_scaled)
        elif model_choice_defect == "Perceptron":
            pred = perceptron_multi.predict(X_scaled)
        else:
            pred = (dnn_multi.predict(X_scaled) > 0.5).astype(int)

        result = dict(zip(label_columns, pred[0]))
        st.write("Predicted Defect Types:")
        st.json(result)
    else:
        st.write("Please enter some text to predict.")

# Add UI requirements for batch prediction
st.write("Upload a CSV file with defect reports for batch prediction (optional)")
uploaded_file = st.file_uploader("Upload CSV for Defect Prediction", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X_batch = vectorizer.transform(df.iloc[:, 0]).toarray()
    X_batch_scaled = scaler_defect.transform(X_batch)
    pred_batch = dnn_multi.predict(X_batch_scaled) > 0.5
    df_result = pd.DataFrame(pred_batch, columns=label_columns)
    st.write("Batch Prediction Results for Defects:")
    st.dataframe(df_result)

# Task 1: Deepfake Audio Detection
st.header("Deepfake Audio Detection")
audio_input = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])
model_choice_deepfake = st.selectbox("Select Model for Deepfake Detection", ["SVM", "Logistic Regression", "Perceptron", "DNN"])

if st.button("Analyze Audio"):
    if audio_input:
        # Save uploaded audio temporarily
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_input.getbuffer())
        
        # Extract features and predict
        X_audio = extract_mfcc(audio_path).reshape(1, -1)
        X_audio_scaled = scaler_audio.transform(X_audio)

        if model_choice_deepfake == "SVM":
            pred = svm_deepfake.predict(X_audio_scaled)[0]
            conf = svm_deepfake.predict_proba(X_audio_scaled)[0][pred]
        elif model_choice_deepfake == "Logistic Regression":
            pred = lr_deepfake.predict(X_audio_scaled)[0]
            conf = lr_deepfake.predict_proba(X_audio_scaled)[0][pred]
        elif model_choice_deepfake == "Perceptron":
            pred = perceptron_deepfake.predict(X_audio_scaled)[0]
            conf = None
        else:
            pred = (dnn_deepfake.predict(X_audio_scaled) > 0.5).astype(int)[0]
            conf = dnn_deepfake.predict(X_audio_scaled)[0][0]

        label = "Bonafide" if pred == 0 else "Deepfake"
        confidence = f"{conf:.2%}" if conf is not None else "N/A"

        st.write("Prediction Result:")
        st.write(f"Label: {label}")
        st.write(f"Confidence: {confidence}")
        
        # Clean up temporary file
        os.remove(audio_path)
    else:
        st.write("Please upload an audio file to analyze.")