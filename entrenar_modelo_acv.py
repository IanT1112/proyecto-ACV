# entrenar_modelo_acv.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Cargar dataset
df = pd.read_csv("stroke.csv")

# Opcional: si el dataset real usa nombres distintos, ajusta nombres de columnas
# Aquí asumimos columnas: age, bmi, avg_glucose_level, hypertension, heart_disease, ever_married, smoking_status, gender, stroke

# Eliminar filas con target desconocido o NA
df = df.dropna(subset=["stroke"])

# Selección de características (ajusta según tu dataset real)
features = ["age", "bmi", "avg_glucose_level", "hypertension", "heart_disease",
            "ever_married", "smoking_status", "gender"]
target = "stroke"

# Asegurarse que existan columnas (si usas el CSV demo, ajusta)
for col in features:
    if col not in df.columns:
        raise ValueError(f"Falta la columna esperada en CSV: {col}")

X = df[features].copy()
y = df[target].astype(int)

# Preprocesamiento
numeric_features = ["age", "bmi", "avg_glucose_level"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_features = ["hypertension", "heart_disease", "ever_married", "smoking_status", "gender"]
# Convertimos 'hypertension' y 'heart_disease' a str (porque OneHotEncoder espera cate.)
X["hypertension"] = X["hypertension"].astype(str)
X["heart_disease"] = X["heart_disease"].astype(str)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Pipeline completo
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
])

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar
clf.fit(X_train, y_train)

# Evaluación
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")
except Exception as e:
    print("No se pudo calcular ROC AUC:", e)

# Guardar modelo
joblib.dump(clf, "modelo_acv.pkl")
print("Modelo guardado en modelo_acv.pkl")
