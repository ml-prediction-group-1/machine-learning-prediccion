import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import argparse
import os

def load_data(path):
    """Carga el dataset convertido a CSV desde Croissant."""
    return pd.read_csv(path)

def preprocess(df):
    """
    Preprocesado para predecir el Ranking:
    - Eliminamos nulos
    - Separación en X (features) e y (target)
    - One-hot encoding para columnas categóricas
    """

    # Nos aseguramos de que existe la columna Ranking
    if "Ranking" not in df.columns:
        raise ValueError("No se encuentra la columna 'Ranking' en el dataset.")

    df = df.dropna()

    y = df["Ranking"]
    X = df.drop(["Ranking"], axis=1)

    # Convertir columnas categóricas a numéricas
    X = pd.get_dummies(X)

    return X, y

def train(args):
    df = load_data(args.data)
    X, y = preprocess(df)

    # División entrenamiento / validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluación
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    print(f"MAE en validación: {mae:.4f}")

    # Guardar modelo
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    train(args)
