# app/main.py

import streamlit as st
import pandas as pd

import os
import sys

# --- AÑADIR RAÍZ DEL PROYECTO AL PYTHONPATH ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.model import (
    load_historical_data,
    compute_team_and_league_stats,
    score_fixtures,
)


@st.cache_data(show_spinner=True)
def _load_hist_and_stats():
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


def main():
    st.set_page_config(
        page_title="Selector de partidos – Estrategia HT",
        layout="wide",
    )

    st.sidebar.title("Selector de partidos")
    st.sidebar.write("Versión inicial – modelo basado en históricos Football-Data.")

    # Cargar históricos y stats
    with st.spinner("Cargando históricos y estadísticas de ligas/equipos..."):
        hist, team_stats, div_stats = _load_hist_and_stats()

    st.success("Histórico cargado correctamente.")

    st.header("1. Cargar fixture (Football-Data)")
    st.write("Sube el fichero `fixtures.xlsx` (tal cual lo descargas de football-data.co.uk).")

    uploaded = st.file_uploader("Selecciona el fichero de fixtures", type=["xlsx"])

    if uploaded is None:
        st.info("Sube un fixture para continuar.")
        return

    try:
        fixtures_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error leyendo el Excel: {e}")
        return

    st.subheader("Vista previa del fixture")
    st.dataframe(fixtures_raw.head())

    # Columnas mínimas requeridas
    required_cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    missing = [c for c in required_cols if c not in fixtures_raw.columns]
    if missing:
        st.error(f"Faltan columnas necesarias en el fixture: {missing}")
        return

    st.header("2. Clasificación según el modelo")

    if st.button("Calcular picks"):
        with st.spinner("Calculando scores de partidos..."):
            fixtures_core = fixtures_raw[required_cols].copy()
            scored = score_fixtures(fixtures_core, team_stats, div_stats)

        # Tabla completa con score
        st.subheader("Todos los partidos con score de modelo")
        st.dataframe(
            scored[
                [
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
                    "FavOdd",
                    "NoClearFav",
                    "PickType",
                ]
            ].sort_values(["Date", "Time", "Div", "HomeTeam"])
        )

        st.subheader("Picks recomendados (Ideal / Buena filtrada)")

        picks = scored[scored["PickType"].notna()].copy()
        if picks.empty:
            st.warning("Hoy no hay partidos que cumplan todos los filtros.")
        else:
            st.success(f"Partidos recomendados: {len(picks)}")

            st.dataframe(
                picks[
                    [
                        "Date",
                        "Time",
                        "Div",
                        "HomeTeam",
                        "AwayTeam",
                        "B365H",
                        "B365D",
                        "B365A",
                        "LeagueTier",
                        "MatchScore",
                        "MatchClass",
                        "PickType",
                    ]
                ].sort_values(["Date", "Time", "Div", "HomeTeam"])
            )

            # Descargar CSV
            csv = picks.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar picks como CSV",
                data=csv,
                file_name="picks_modelo.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
