# 🚀 Predicción del Ranking de Universidades de Pakistán  
Repositorio colaborativo para el desarrollo de un sistema reproducible de Machine Learning mediante buenas prácticas de ingeniería, control de versiones y documentación profesional.

---

# 🧠 1. Introducción  
Este proyecto tiene como objetivo construir un servicio reproducible capaz de predecir el **ranking de universidades** pakistaníes utilizando modelos de Machine Learning.

Se simula un entorno profesional de trabajo colaborativo, aplicando:
- Git y GitHub en equipo  
- Issues + Kanban board  
- Protección de rama `main`  
- Pre-commit hooks  
- Documentación interactiva (Jupyter Book)  
- EDA completo  
- Pipeline de entrenamiento y predicción  

El dataset se obtiene mediante **Kaggle Croissant** (JSON-LD), garantizando trazabilidad y reproducibilidad del origen de datos.

---

# 🎯 2. Problema a Resolver  
El objetivo es predecir el **Ranking** de distintas universidades en Pakistán a partir de sus características institucionales.

### ¿Por qué es relevante?
- Permite evaluar instituciones según características comunes.  
- Puede ayudar a estudios académicos, consultoras o análisis de rendimiento educativo.  
- Es un caso realista de regresión supervisada.

---

# 📊 3. Descripción del Dataset  
El dataset original proviene de Kaggle:

🔗 https://www.kaggle.com/datasets/ayeshaseherr/top-pakistani-universities

Se extrae mediante Croissant y contiene información como:
- Nombre  
- Tipo de universidad  
- Provincia  
- Enrollments (tamaño)  
- Ranking (variable objetivo)

Tras el EDA, se genera un dataset limpio en:


---

# 🔍 4. Exploratory Data Analysis (EDA)

El EDA se encuentra en:

📁 `notebooks/01_exploracion.ipynb`

Incluye:
- Distribuciones de variables  
- Valores nulos  
- Correlaciones  
- Outliers  
- Limpieza final  
- Exportación del CSV preparado  

---

# 🧩 5. Pipeline del Proyecto (Mermaid)

```mermaid
flowchart TD
    A[Kaggle Dataset<br>(Croissant JSON-LD)] --> B[EDA<br>01_exploracion.ipynb]
    B --> C[data/universities.csv]
    C --> D[Entrenamiento<br>training.py]
    D --> E[Modelo<br>model.joblib]
    E --> F[Predicción<br>prediction.py]
    F --> G[preds.csv]

📌 Nota técnica: Se añade esta línea para forzar la ejecución del workflow de pre-commit en GitHub Actions.
