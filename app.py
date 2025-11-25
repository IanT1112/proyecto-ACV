# app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Sistema IA de Detección Temprana de ACV", page_icon="🧠", layout="centered")

st.title("Sistema IA de Detección Temprana de ACV")
st.markdown(
    """
Este sistema es **únicamente demostrativo**. No reemplaza una evaluación médica profesional.
Rellena los datos y pulsa **Calcular Riesgo**.
"""
)

# --- Formulario ---
with st.form("formulario_datos"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad (años)", min_value=1, max_value=120, value=50)
        bmi = st.number_input("IMC (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        avg_glucose_level = st.number_input("Glucosa promedio (mg/dL)", min_value=40.0, max_value=400.0, value=100.0, step=0.1)
    with col2:
        hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
        heart_disease = st.selectbox("Enfermedad cardiaca", ["No", "Sí"])
        smoking_status = st.selectbox("Estado de fumador", ["nunca fumé", "anteriormente fumé", "fumador", "desconocido"])
    gender = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
    submitted = st.form_submit_button("Calcular Riesgo")

if submitted:
    # Cargar modelo (pipeline)
    try:
        clf = joblib.load("modelo_acv.pkl")
    except Exception as e:
        st.error("No se encontró el modelo entrenado. Ejecuta entrenar_modelo_acv.py primero.")
        st.stop()

    # Preparar entrada como DataFrame con las mismas columnas que uso el script de entrenamiento
    entrada = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "avg_glucose_level": avg_glucose_level,
        "hypertension": str(1 if hypertension == "Sí" else 0),
        "heart_disease": str(1 if heart_disease == "Sí" else 0),
        "ever_married": "Yes",   # si quieres pedirlo en formulario, agrégalo; por ahora dejamos Yes
        "smoking_status": smoking_status,
        "gender": gender
    }])

    # Predecir probabilidad
    try:
        prob = clf.predict_proba(entrada)[0][1]  # probabilidad de clase 1 (stroke)
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.stop()

    riesgo_pct = prob * 100

    if riesgo_pct < 30:
        st.success(f"Riesgo estimado de ACV: Bajo — {riesgo_pct:.2f}%")
    elif riesgo_pct < 60:
        st.warning(f"Riesgo estimado de ACV: Moderado — {riesgo_pct:.2f}%")
    else:
        st.error(f"Riesgo estimado de ACV: Alto — {riesgo_pct:.2f}%")

    st.markdown("---")
    st.write("**Interpretación**: Este resultado es sólo una estimación basada en el modelo. Consulte un profesional de la salud para una evaluación clínica.")
