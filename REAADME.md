# Sistema IA de Detección Temprana de ACV

**Proyecto demostrativo** del curso *Aprendizaje Estadístico* que implementa un clasificador para estimar el riesgo de accidente cerebrovascular (ACV).

> ⚠ Este sistema es únicamente educativo. No sustituye una evaluación médica.

## Contenido del repositorio
- `app.py` : App web con Streamlit (formulario + predicción).
- `entrenar_modelo_acv.py` : Script para entrenar y guardar `modelo_acv.pkl`.
- `modelo_acv.pkl` : Modelo entrenado (se genera tras ejecutar el script).
- `stroke.csv` : Dataset usado (debe colocarse en la carpeta).
- `requirements.txt` : Dependencias.

## Instrucciones de uso

### 1. Preparar entorno
```bash
python -m venv venv     
pip install -r requirements.txt

python entrenar_modelo_acv.py
# verifica que se generó modelo_acv.pkl


python -m streamlit run app.py
# abrir http://localhost:8501

