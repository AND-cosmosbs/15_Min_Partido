# app/main.py

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --- Añadir raíz del proyecto al PYTHONPATH ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.model import (  # type: ignore
    load_historical_data,
    compute_team_and_league_stats,
    score_fixtures,
)


# ---------- CARGA HISTÓRICO (CACHEADO) ----------

@st.cache_data(show_spinner="Cargando histórico y calculando estadísticas…")
def _load_hist_and_stats():
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


# ---------- INTERFAZ PRINCIPAL ----------

def main():
    st.set_page_config(
        page_title="Selector de Partidos HT/FT",
        layout="wide",
    )

    st.title("Selector de partidos HT/FT – Modelo")

    # Cargar histórico + stats
    try:
        hist, team_stats, div_stats = _load_hist_and_stats()
        st.success("Histórico cargado correctamente.")
    except Exception as e:
        st.error(f"Error cargando histórico: {e}")
        return

    st.markdown("### 1. Cargar fixture (Football-Data)")

    uploaded = st.file_uploader(
        "Sube el fichero `fixtures.xlsx` (tal cual de football-data.co.uk)",
        type=["xlsx", "xls"],
    )

    if uploaded is None:
        st.info("Sube un fixture para continuar.")
        return

    # Leemos el fichero subido
    try:
        fixtures_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error leyendo el fichero de fixtures: {e}")
        return

    # Nos quedamos con las columnas clave
    expected_cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam",
                     "B365H", "B365D", "B365A"]
    missing = [c for c in expected_cols if c not in fixtures_raw.columns]
    if missing:
        st.error(f"Faltan columnas en el fixture: {missing}")
        return

    fixtures_df = fixtures_raw[expected_cols].copy()

    # ---------- Aplicar modelo ----------
    try:
        scored = score_fixtures(team_stats, div_stats, fixtures_df)
    except Exception as e:
        st.error(f"Error aplicando el modelo a los fixtures: {e}")
        return

    # Filtrar solo partidos con PickType (Ideal / Buena filtrada)
    picks = scored[scored["PickType"].notna()].copy()

    st.markdown("### 2. Resultados del modelo")

    if picks.empty:
        st.warning("Ningún partido cumple los filtros del modelo para este fixture.")
        return

    # Tabla reducida (lo que me pediste)
    cols_to_show = [
        "Date",
        "Time",
        "Div",
        "HomeTeam",
        "AwayTeam",
        "B365H",
        "B365D",
        "B365A",
        "L_score",
        "LeagueTier",
        "H_T_score",
        "A_T_score",
        "MatchScore",
        "MatchClass",
        "PickType",
    ]

    table = picks[cols_to_show].sort_values(["Date", "Time", "Div", "HomeTeam"])

    st.dataframe(
        table,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
