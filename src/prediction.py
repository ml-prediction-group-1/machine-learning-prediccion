import pandas as pd
import joblib
import argparse

def preprocess_new_data(df):
    df = pd.get_dummies(df)
    return df

def predict(model_path, input_path, output_path):
    # Cargar modelo
    model = joblib.load(model_path)

    # Cargar datos
    X = pd.read_csv(input_path)
    X = preprocess_new_data(X)

    # Asegurar columnas iguales a entrenamiento
    # (En caso de faltantes, se rellenan con 0)
    model_features = model.feature_names_in_
    X = X.reindex(columns=model_features, fill_value=0)

    preds = model.predict(X)

    pd.DataFrame({"prediction_ranking": preds}).to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predicciones.csv")
    args = parser.parse_args()

    predict(args.model, args.input, args.output)

