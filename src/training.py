import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import argparse
import os
import sys

# ============================================================
# 1. CONTROL DE DETERMINISMO
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. CARGA DEL DATASET
# ============================================================

def load_data(path):
    """Carga el dataset desde un CSV validando su existencia."""
    if not os.path.exists(path):
        print(f"[ERROR] No se encontró el dataset en la ruta: {path}")
        sys.exit(1)

    print(f"[INFO] Cargando dataset desde: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset cargado correctamente ({df.shape[0]} filas, {df.shape[1]} columnas)")
    return df

# ============================================================
# 3. PREPROCESAMIENTO
# ============================================================

def preprocess(df):
    """
    Preprocesado:
    - Eliminación de nulos
    - One-hot encoding
    - Separación en X (features) e y (target)
    """

    if "Ranking" not in df.columns:
        raise ValueError("[ERROR] La columna 'Ranking' no existe en el dataset.")

    # Eliminamos nulos
    df = df.dropna()
    print(f"[INFO] Filas tras eliminar nulos: {df.shape[0]}")

    # Target
    y = df["Ranking"]

    # Features
    X = df.drop(columns=["Ranking"])

    # One-hot encoding para variables categóricas
    X = pd.get_dummies(X, drop_first=True)
    print(f"[INFO] Features generadas tras encoding: {X.shape[1]}")
    
    return X, y

# ============================================================
# 4. ENTRENAMIENTO DEL MODELO
# ============================================================

def train(args):
    df = load_data(args.data)
    X, y = preprocess(df)

    print("[INFO] Dividiendo dataset en entrenamiento y validación...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    print("[INFO] Entrenando modelo RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=SEED
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    print(f"[RESULTADO] MAE en validación: {mae:.4f}")

    # Guardar modelo
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"[INFO] Modelo guardado correctamente en: {model_path}")

# ============================================================
# 5. EJECUCIÓN DIRECTA DEL SCRIPT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de entrenamiento del modelo ML")
    parser.add_argument("--data", required=True, help="Ruta al CSV del dataset")
    parser.add_argument("--out_dir", default="models", help="Directorio donde guardar el modelo")

    args = parser.parse_args()

    train(args)

