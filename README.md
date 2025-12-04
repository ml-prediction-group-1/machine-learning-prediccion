# Práctica Tema 1 - Machine Learning

Repositorio para el desarrollo de un modelo de predicción reproducible usando buenas prácticas de Git y GitHub.

## Estructura del proyecto
- notebooks/: análisis y experimentación
- src/: código final de entrenamiento y predicción
- data/: datos (o enlaces a ellos)
- models/: modelos entrenados

## Ejecución
1. Instalar dependencias:
pip install -r requirements.txt

2. Entrenar:
python src/training.py --data data/dataset.csv --out_dir models

3. Predecir:
python src/prediction.py --model models/model.joblib --input data/test.csv --output preds.csv
