import streamlit as st
import pandas as pd

from backend.model import load_historical_data, compute_team_and_league_stats, score_fixtures


st.set_page_config(
    page_title="Estrategia HT/FT – Selector de partidos",
    layout="wide",
)

# Sidebar (el desplegable ya lo tienes, aquí lo usamos)
st.sidebar.title("Menú")
page = st.sidebar.selectbox(
    "Selecciona sección",
    ["Clasificar fixtures", "Seguimiento (próximamente)"]
)

st.title("Estrategia HT/FT – Partidos con baja probabilidad de gol temprano")


@st.cache_data(show_spinner=True)
def get_model_stats():
    data = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(data)
    return team_stats, div_stats


if page == "Clasificar fixtures":
    st.subheader("1. Subir fixtures (formato Football-Data)")

    uploaded = st.file_uploader(
        "Sube el archivo fixtures.xlsx o similar",
        type=["xlsx", "xls", "csv"]
    )

    if uploaded is not None:
        # Detectar tipo
        if uploaded.name.endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)

        st.write("Vista previa del archivo subido:")
        st.dataframe(raw.head())

        # Intentar quedarnos con las columnas estándar
        expected_cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
        missing = [c for c in expected_cols if c not in raw.columns]

        if missing:
            st.error(f"Faltan columnas en el fixture: {missing}")
        else:
            # Normalizar fecha
            fixtures = raw[expected_cols].copy()
            fixtures["Date"] = pd.to_datetime(
                fixtures["Date"], dayfirst=True, errors="coerce"
            )

            st.subheader("2. Calcular puntuación de partidos")

            with st.spinner("Calculando modelo a partir del histórico..."):
                team_stats, div_stats = get_model_stats()
                scored = score_fixtures(fixtures, team_stats, div_stats)

            # Mostrar solo columnas relevantes
            cols_show = [
                "Date", "Time", "Div", "HomeTeam", "AwayTeam",
                "B365H", "B365D", "B365A",
                "LeagueTier", "L_score", "H_T_score", "A_T_score",
                "MatchScore", "MatchClass", "PickType"
            ]
            scored_show = scored[cols_show].sort_values(["Date", "Time", "Div", "HomeTeam"])

            st.subheader("Resultados clasificados")
            st.dataframe(scored_show, use_container_width=True)

            # Filtro rápido: solo picks a apostar
            picks_only = scored_show[scored_show["PickType"].notna()]
            st.markdown("### Partidos seleccionados (Ideal / Buena filtrada)")
            st.dataframe(picks_only, use_container_width=True)

            # Botón para descargar CSV
            csv = scored_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Descargar CSV clasificado",
                data=csv,
                file_name="fixtures_clasificados.csv",
                mime="text/csv",
            )

elif page == "Seguimiento (próximamente)":
    st.info("La sección de seguimiento de partidos la montamos en el siguiente paso (tracking de ROI, minuto de cierre, etc.).")
