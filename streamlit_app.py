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
