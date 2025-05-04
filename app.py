import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models and preprocessing objects
lr_multi = joblib.load('lr_multi_defect.pkl')
svm_multi = joblib.load('svm_multi_defect.pkl')
perceptron_multi = joblib.load('perceptron_multi_defect.pkl')
dnn_multi = tf.keras.models.load_model('dnn_multi_defect.h5')
scaler = joblib.load('scaler_defect.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit UI
st.title("Software Defect Prediction App")

# Defect Prediction
st.header("Defect Prediction")
text_input = st.text_area("Enter defect report text", height=100)
model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM", "Perceptron", "DNN"])

if st.button("Predict"):
    if text_input:
        # Transform input text using TF-IDF
        X = vectorizer.transform([text_input]).toarray()
        X_scaled = scaler.transform(X)

        if model_choice == "Logistic Regression":
            pred = lr_multi.predict(X_scaled)
        elif model_choice == "SVM":
            pred = svm_multi.predict(X_scaled)
        elif model_choice == "Perceptron":
            pred = perceptron_multi.predict(X_scaled)
        else:
            pred = (dnn_multi.predict(X_scaled) > 0.5).astype(int)

        label_columns = ['type_blocker', 'type_regression', 'type_bug', 'type_documentation', 
                        'type_enhancement', 'type_task', 'type_dependency_upgrade']
        result = dict(zip(label_columns, pred[0]))
        st.write("Predicted Defect Types:")
        st.json(result)
    else:
        st.write("Please enter some text to predict.")

# Add UI requirements
st.write("Upload a CSV file with defect reports for batch prediction (optional)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X_batch = vectorizer.transform(df.iloc[:, 0]).toarray()
    X_batch_scaled = scaler.transform(X_batch)
    pred_batch = dnn_multi.predict(X_batch_scaled) > 0.5
    df_result = pd.DataFrame(pred_batch, columns=label_columns)
    st.write("Batch Prediction Results:")
    st.dataframe(df_result)