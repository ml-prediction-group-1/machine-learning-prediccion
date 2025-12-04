import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import argparse
import os

def load_data(path):
    """Carga el dataset desde un CSV"""
    return pd.read_csv(path)

def preprocess(df):
    """Ejemplo de preprocesado simple"""
    df = df.dropna()
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def train(args):
    df = load_data(args.data)
    X, y = preprocess(df)

    # División train/validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo simple
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Accuracy de validación: {acc:.4f}")

    # Guardar modelo
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, "model.joblib")
    joblib.dump(model, output_path)

    print(f"Modelo guardado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Ruta al CSV con columna 'target'")
    parser.add_argument("--out_dir", default="models", help="Carpeta donde guardar el modelo")

    args = parser.parse_args()
    train(args)

