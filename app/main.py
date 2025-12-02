import streamlit as st
import pandas as pd

st.set_page_config(page_title="Estrategia HT/FT", layout="wide")

st.title("Estrategia 15 minutos")

st.sidebar.header("Navegación")
page = st.sidebar.radio(
    "Selecciona sección",
    ["Resumen", "Cargar fixtures (prueba)", "Seguimiento (placeholder)"]
)

if page == "Resumen":
    st.subheader("Resumen del modelo")
    st.markdown("""
    - Modelo basado en histórico de 3 temporadas (Football-Data).
    - Clasificación de ligas y equipos por probabilidad de 0-0 al descanso.
    - Filtro por cuotas para evitar favoritos claros.
    - Objetivo: encontrar **partidos Ideal** (y más adelante **Buena filtrada**).
    """)

elif page == "Cargar fixtures (prueba)":
    st.subheader("Cargar fixtures (modo demo)")

    uploaded = st.file_uploader(
        "Sube el Excel de fixtures (el mismo formato de Football-Data)",
        type=["xls", "xlsx"]
    )

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
            st.success(f"Archivo cargado con {len(df)} filas")
            st.dataframe(df.head(20))
        except Exception as e:
            st.error(f"Error leyendo el Excel: {e}")

elif page == "Seguimiento (placeholder)":
    st.subheader("Seguimiento de apuestas (placeholder)")
    st.info("Aquí mostraremos el fichero de seguimiento con ROI por partido, etc., en la siguiente iteración.")
