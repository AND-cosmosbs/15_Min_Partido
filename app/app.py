import streamlit as st

st.set_page_config(page_title="Match Analyzer", layout="wide")

st.title("⚽ Match Analyzer – v0.1")

st.write("""
Bienvenido.  
Esta es la versión inicial de la app donde cargaremos fixtures, 
aplicaremos filtros y clasificaremos los partidos (Ideal, Buena filtrada, etc.).
""")

st.subheader("Carga de fixture")
uploaded_file = st.file_uploader("Sube el archivo de fixtures (.xlsx)")

if uploaded_file:
    import pandas as pd
    df = pd.read_excel(uploaded_file)
    st.write("Primeras filas del archivo cargado:")
    st.dataframe(df.head())
