import streamlit as st
import numpy as np

st.title("Simulador de Predicción")

edad = st.number_input("Edad", min_value=0, max_value=120)
ingresos = st.number_input("Ingresos (€)", min_value=0.0)
genero = st.selectbox("Género", ["Hombre", "Mujer"])
score = st.slider("Score", 0.0, 1.0, 0.5)

# Validación
if edad <= 0:
    st.error("Edad inválida") 

if st.button("Predecir"):
    pred = edad * 0.1 + ingresos * 0.01 + (1 if genero=="Hombre" else 0) + score * 5 
    st.success(f"Predicción estimada: {pred:.2f}")
import matplotlib.pyplot as plt
import pandas as pd

st.header("Métricas históricas del modelo (simuladas)")

# Fechas simuladas
fechas = pd.date_range("2024-01-01", periods=12, freq="M")

# Métricas simuladas
accuracy = np.random.uniform(0.7, 0.9, 12)
f1 = np.random.uniform(0.6, 0.85, 12)
loss = np.random.uniform(0.3, 0.7, 12)

# Gráfico: Accuracy
fig1, ax1 = plt.subplots()
ax1.plot(fechas, accuracy, marker="o")
ax1.set_title("Accuracy histórico")
ax1.set_ylabel("Accuracy")
ax1.grid(True)
st.pyplot(fig1)

# Gráfico: F1-score
fig2, ax2 = plt.subplots()
ax2.plot(fechas, f1, marker="o", color="green")
ax2.set_title("F1-score histórico")
ax2.set_ylabel("F1-score")
ax2.grid(True)
st.pyplot(fig2)

# Gráfico: Loss
fig3, ax3 = plt.subplots()
ax3.plot(fechas, loss, marker="o", color="red")
ax3.set_title("Loss histórico")
ax3.set_ylabel("Loss")
ax3.grid(True)
st.pyplot(fig3)

