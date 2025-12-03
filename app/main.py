import os
import sys

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------
# AÑADIR RAÍZ DEL PROYECTO AL PYTHONPATH PARA PODER IMPORTAR backend
# -------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from backend.model import (
    load_historical_data,
    compute_team_and_league_stats,
    score_fixtures,
)

# -------------------------------------------------------------------
# CARGA DE HISTÓRICO + ESTADÍSTICAS (CACHEADO)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def _load_hist_and_stats():
    """
    Carga todos los all-euro-data-*.xls* desde /data
    y calcula estadísticas de ligas y equipos.
    """
    hist = load_historical_data("data")  # busca en /opt/render/project/src/data
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


# -------------------------------------------------------------------
# APLICACIÓN PRINCIPAL
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Selector de Partidos HT Estrategia",
        layout="wide",
    )

    st.title("Selector de partidos – Estrategia HT (BTTS NO + U3.5 + 1-1)")
    st.markdown(
        """
Esta app:

1. Carga el histórico de **Football-Data** (all-euro-data-*.xls*) desde la carpeta `/data`.
2. Calcula perfiles de **ligas** y **equipos**.
3. Clasifica los partidos del **fixtures.xlsx** en:
   - `Ideal`
   - `Buena`
   - `Borderline`
   - `Descartar`
4. Aplica los filtros extra que definimos:
   - Sin favorito claro (**min(B365H, B365A) ≥ 2.0**).
   - Marca los partidos seleccionados como:
     - `Ideal`
     - `Buena filtrada`
        """
    )

    # ---------------------------
    # CARGA DEL HISTÓRICO
    # ---------------------------
    with st.spinner("Cargando histórico y calculando estadísticas..."):
        hist, team_stats, div_stats = _load_hist_and_stats()

    st.success("Histórico cargado correctamente.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Partidos históricos", f"{len(hist):,}".replace(",", "."))
    with col2:
        st.metric("Equipos", f"{team_stats['Team'].nunique():,}".replace(",", "."))
    with col3:
        st.metric("Ligas", f"{div_stats['Div'].nunique():,}".replace(",", "."))

    st.markdown("---")

    # ---------------------------
    # SUBIR FIXTURES
    # ---------------------------
    st.header("1. Cargar fixture (Football-Data)")

    st.markdown(
        """
Sube el fichero `fixtures.xlsx` **tal cual** lo descargas de football-data.co.uk
(sin tocar columnas ni nombre de hojas).
        """
    )

    uploaded = st.file_uploader(
        "Selecciona el fichero de fixtures",
        type=["xlsx", "xls", "xlsm"],
        key="fixtures_uploader",
    )

    if uploaded is None:
        st.info("Sube un fixture para continuar.")
        return

    # ---------------------------
    # LEER FIXTURE Y CLASIFICAR
    # ---------------------------
    try:
        fixtures_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error leyendo el fichero: {e}")
        return

    # Normalizar nombres clave por si hubiera mayúsculas/minúsculas
    fixtures_df.columns = [str(c).strip() for c in fixtures_df.columns]

    required_cols = {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"}
    missing = required_cols - set(fixtures_df.columns)
    if missing:
        st.error(
            "El fixtures no tiene las columnas mínimas necesarias:\n\n"
            + ", ".join(sorted(missing))
        )
        return

    st.success(f"Fixture cargado: {len(fixtures_df)} partidos.")

    # Clasificación con el modelo
    with st.spinner("Clasificando partidos según el modelo..."):
        scored = score_fixtures(hist, team_stats, div_stats, fixtures_df)

    # Esperamos que score_fixtures devuelva al menos:
    #  - L_score, LeagueTier
    #  - H_T_score, A_T_score
    #  - MatchScore, MatchClass
    #  - FavOdd, NoClearFav
    #  - IsBuenaFiltrada, PickType

    st.header("2. Resultados del modelo")

    # Resumen rápido
    total = len(scored)
    n_ideal = (scored["PickType"] == "Ideal").sum()
    n_buena_f = (scored["PickType"] == "Buena filtrada").sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Partidos en el fixture", total)
    with c2:
        st.metric("Selecciones Ideal", n_ideal)
    with c3:
        st.metric("Selecciones Buena filtrada", n_buena_f)

    st.markdown("### 2.1. Solo partidos seleccionados (Ideal / Buena filtrada)")

    picks = scored[scored["PickType"].notna()].copy()
    if picks.empty:
        st.warning("Hoy no hay ningún partido que cumpla todos los filtros.")
    else:
        # Orden lógico por fecha/hora/liga
        if "Date" in picks.columns:
            # Si Date es texto, intentar parsear para ordenar bien
            if not pd.api.types.is_datetime64_any_dtype(picks["Date"]):
                picks["Date"] = pd.to_datetime(picks["Date"], errors="coerce")
            picks = picks.sort_values(["Date", "Time", "Div", "HomeTeam"])

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
        cols_to_show = [c for c in cols_to_show if c in picks.columns]

        st.dataframe(picks[cols_to_show], use_container_width=True)

    st.markdown("### 2.2. Tabla completa de partidos (diagnóstico)")

    with st.expander("Ver todos los partidos del fixture con su puntuación", expanded=False):
        st.dataframe(scored, use_container_width=True)


if __name__ == "__main__":
    main()
