import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="cinnamon_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Kualitas Kayu Manis")
st.write("Masukkan parameter uji laboratorium untuk memprediksi kualitas kayu manis (baik/buruk).")

# Form input pengguna
moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=20.0, value=11.6, step=0.01)
ash = st.number_input("Ash (%)", min_value=0.0, max_value=10.0, value=6.5, step=0.01)
volatile_oil = st.number_input("Volatile Oil (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.01)
acid_insoluble_ash = st.number_input("Acid Insoluble Ash (%)", min_value=0.0, max_value=2.0, value=0.45, step=0.01)
chromium = st.number_input("Chromium (mg/kg)", min_value=0.0, max_value=0.01, value=0.002, step=0.0001, format="%.4f")
coumarin = st.number_input("Coumarin (mg/kg)", min_value=0.0, max_value=0.05, value=0.008, step=0.0001, format="%.4f")

if st.button("Prediksi Kualitas"):
    input_data = np.array([[moisture, ash, volatile_oil, acid_insoluble_ash, chromium, coumarin]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = int(prediction[0][0] > 0.5) 
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    st.success(f"Kualitas kayu manis diprediksi: **{predicted_label.upper()}**")
