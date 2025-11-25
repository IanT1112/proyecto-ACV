Sistema de IA para la Detección Temprana de ACV
Proyecto de Aprendizaje Estadístico – 2025

DESCRIPCIÓN DEL PROYECTO
Este proyecto implementa un sistema predictivo para la detección temprana de Accidente Cerebrovascular (ACV) utilizando técnicas de aprendizaje estadístico.
El sistema permite entrenar un modelo con datos reales del dataset stroke.csv y luego realizar predicciones introduciendo valores manualmente mediante una aplicación web desarrollada en Streamlit.
El objetivo principal es facilitar la identificación temprana del riesgo de ACV utilizando modelos de machine learning.

ESTRUCTURA DEL PROYECTO

proyecto-ACV/
 | app.py (Aplicación web Streamlit)
 | entrenar_modelo_acv.py (Script para entrenar el modelo)
 | modelo_acv.pkl (Modelo entrenado)
 | stroke.csv (Dataset utilizado)
 | requirements.txt (Dependencias)
 | README.md (Documentación del proyecto)

DATASET UTILIZADO
Se utiliza el dataset "Stroke Prediction Dataset" proveniente de Kaggle.
Las variables disponibles incluyen: edad, género, hipertensión, enfermedad cardiaca, glucosa, IMC, estado fumador, tipo de trabajo, residencia, entre otras.
La variable objetivo es "stroke", donde 1 significa riesgo de ACV y 0 significa sin riesgo.

Instalar dependencias:
pip install -r requirements.txt

ENTRENAMIENTO DEL MODELO
Para entrenar el modelo ejecutar:
python entrenar_modelo_acv.py
Esto generará el archivo modelo_acv.pkl que es utilizado por la aplicación web.

EJECUCIÓN DE LA APLICACIÓN
Para ejecutar la interfaz web:
python -m streamlit run app.py
La aplicación estará disponible en:
http://localhost:8501

FUNCIONALIDAD DE LA APLICACIÓN
La aplicación permite ingresar valores como edad, glucosa, IMC, hipertensión, enfermedad cardiaca, género, tipo de trabajo y estado fumador.
El sistema procesa los datos y devuelve una predicción del riesgo de ACV:

- Riesgo Alto
- Riesgo Bajo

MODELO UTILIZADO
El modelo implementado está basado en técnicas de clasificación, comúnmente Random Forest o Regresión Logística.
Se utilizan prácticas básicas de aprendizaje estadístico:
  - Limpieza de datos
  - Divisiones de entrenamiento y prueba
  - Evaluación con Accuracy, Recall y F1-score

DEPLOY
La aplicación puede desplegarse en plataformas como Streamlit Cloud.
Solo se debe subir el repositorio a GitHub y crear un nuevo deploy desde la plataforma.

TECNOLOGÍAS UTILIZADAS
  - Python 3.11
  - Pandas
  - Numpy
  - Scikit-learn
  - Streamlit
  - Git y GitHub

Estudiante de Ingeniería de Sistemas e Inteligencia Artificial
2025
