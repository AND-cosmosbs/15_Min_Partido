import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------
# CONFIGURAR RUTAS PARA PODER IMPORTAR backend.model
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # carpeta raíz del repo
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.model import (
    load_historical_data,
    compute_team_and_league_stats,
    score_fixtures,
)

# -------------------------------------------------------------------
# CACHÉ PARA HISTÓRICO Y STATS
# -------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando histórico y estadísticas...")
def _load_hist_and_stats():
    # La carpeta 'data' está en la raíz del proyecto (al lado de app/, backend/, etc.)
    data_dir = ROOT_DIR / "data"
    hist = load_historical_data(data_dir)
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


# -------------------------------------------------------------------
# INTERFAZ
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Selector de Partidos HT (modelo 0-0)",
        layout="wide",
    )

    st.title("Selector de partidos – Estrategia 0-0 HT / BTTS NO / Under 3.5 / 1-1")

    # Cargar histórico + stats
    try:
        hist, team_stats, div_stats = _load_hist_and_stats()
        st.success("Histórico cargado correctamente.")
    except Exception as e:
        st.error(f"Error cargando histórico: {e}")
        st.stop()

    st.markdown("### 1. Cargar fixture (Football-Data)")
    st.write(
        "Sube el fichero `fixtures.xlsx` tal cual lo descargas de football-data.co.uk."
    )

    uploaded = st.file_uploader(
        "Selecciona el fichero de fixtures",
        type=["xlsx", "xlsm", "xls"],
    )

    if uploaded is None:
        st.info("Sube un fixture para continuar.")
        return

    # ----------------------------------------------------------------
    # LEER FIXTURE SUBIDO
    # ----------------------------------------------------------------
    try:
        fixtures_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"No se ha podido leer el Excel subido: {e}")
        return

    # Normalizar nombres de columnas esperadas
    expected_cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    missing = [c for c in expected_cols if c not in fixtures_df.columns]
    if missing:
        st.error(
            "El fichero de fixtures no tiene las columnas esperadas.\n"
            f"Faltan: {missing}"
        )
        return

    # Convertir fecha y hora
    fixtures_df["Date"] = pd.to_datetime(
        fixtures_df["Date"], dayfirst=True, errors="coerce"
    )

    # ----------------------------------------------------------------
    # APLICAR MODELO
    # score_fixtures(team_stats, div_stats, fixtures_df)
    # ----------------------------------------------------------------
    try:
        scored = score_fixtures(team_stats, div_stats, fixtures_df)
    except Exception as e:
        st.error(f"Error aplicando el modelo a los fixtures: {e}")
        return

    # ----------------------------------------------------------------
    # FILTRO Y SALIDA: solo columnas importantes
    # ----------------------------------------------------------------
    st.markdown("### 2. Resultados del modelo")

    # Si tu score_fixtures ya añade PickType / MatchClass / L_score, etc.,
    # filtramos a lo relevante:
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
    cols_present = [c for c in cols_to_show if c in scored.columns]

    if not cols_present:
        st.error(
            "El DataFrame resultante no contiene las columnas esperadas "
            "(L_score, H_T_score, A_T_score, MatchScore, MatchClass, PickType...)."
        )
        st.dataframe(scored)
        return

    # Ordenar por fecha/hora
    if "Date" in scored.columns and "Time" in scored.columns:
        scored = scored.sort_values(["Date", "Time", "Div", "HomeTeam"])

    # Mostrar tabla filtrada
    st.dataframe(scored[cols_present], use_container_width=True)

    # Un pequeño resumen por tipo de pick
    if "PickType" in scored.columns:
        st.markdown("#### Resumen por tipo de pick")
        st.write(scored["PickType"].value_counts())


if __name__ == "__main__":
    main()
