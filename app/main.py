# app/main.py

import os
import sys

import streamlit as st
import pandas as pd

# --- AÑADIR RAÍZ DEL PROYECTO AL PYTHONPATH ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# --- IMPORTAR LÓGICA DEL MODELO ---
from backend.model import (
    load_historical_data,
    compute_team_and_league_stats,
    score_fixtures,
)


# ---------- CARGA DEL HISTÓRICO (CACHEADO) ----------
@st.cache_data(show_spinner=True)
def _load_hist_and_stats():
    """
    Carga los históricos de /data, calcula stats de equipos y ligas
    y los devuelve cacheados.
    """
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


# ---------- APP PRINCIPAL ----------
def main():
    st.set_page_config(
        page_title="Selector de partidos HT/FT",
        layout="wide",
    )

    st.title("Selector de partidos – Estrategia HT/FT 0–0 al descanso")
    st.markdown(
        """
        Esta app:
        1. Usa tu histórico (3 temporadas).
        2. Calcula perfil de ligas y equipos.
        3. Clasifica los partidos del fixture en: **Ideal / Buena / Borderline / Descartar**.
        4. Aplica el filtro de **no favorito claro** (min(B365H, B365A) ≥ 2.0).
        """
    )

    # --- Cargar histórico y stats ---
    with st.spinner("Cargando histórico y calculando estadísticas de ligas/equipos..."):
        hist, team_stats, div_stats = _load_hist_and_stats()
    st.success("Histórico cargado correctamente.")

    st.markdown("### 1. Cargar fixture (Football-Data)")

    uploaded_file = st.file_uploader(
        "Sube el fichero `fixtures.xlsx` tal cual lo descargas de football-data.co.uk",
        type=["xlsx", "xls"],
    )

    if uploaded_file is None:
        st.info("Sube un fixture para continuar.")
        return

    # --- Leer fixture subido ---
    try:
        fixtures_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error leyendo el fichero de fixtures: {e}")
        return

    if fixtures_df.empty:
        st.warning("El fixture está vacío o no se ha podido leer correctamente.")
        return

    st.success("Fixture cargado. Procesando partidos con el modelo...")

    # --- Aplicar modelo y filtros (Ideal / Buena filtrada / etc.) ---
    try:
        scored = score_fixtures(
            fixtures_df,
            team_stats=team_stats,
            div_stats=div_stats,
            fav_threshold=2.0,       # min(B365H,B365A) ≥ 2.0
            include_buena_filtrada=True,
        )
    except TypeError:
        # Por si tu model.score_fixtures no acepta todos los kwargs
        scored = score_fixtures(fixtures_df, team_stats, div_stats)

    if scored.empty:
        st.warning("No se han podido puntuar los partidos del fixture.")
        return

    # Esperamos estas columnas de salida desde score_fixtures:
    # Div, Date, Time, HomeTeam, AwayTeam, B365H, B365D, B365A,
    # L_score, LeagueTier, H_T_score, A_T_score, MatchScore, MatchClass, PickType
    st.markdown("### 2. Resultados del modelo")

    # Vista que quiere ver el usuario
    view = st.radio(
        "¿Qué quieres ver?",
        ("Solo partidos a apostar", "Todos los partidos puntuados"),
        index=0,
        horizontal=True,
    )

    if "PickType" in scored.columns:
        picks = scored[scored["PickType"].notna()].copy()
    else:
        picks = scored.iloc[0:0].copy()  # tabla vacía con mismas columnas

    if view == "Solo partidos a apostar":
        df_show = picks
        st.subheader("Partidos seleccionados (Ideal / Buena filtrada)")
    else:
        df_show = scored
        st.subheader("Todos los partidos puntuados")

    if df_show.empty:
        st.info("No hay partidos que cumplan los criterios para hoy con el fixture subido.")
        return

    # Ordenar un poco
    for col in ["Date", "Time"]:
        if col in df_show.columns:
            df_show[col] = pd.to_datetime(df_show[col], errors="coerce")

    sort_cols = [c for c in ["Date", "Time", "Div", "HomeTeam"] if c in df_show.columns]
    if sort_cols:
        df_show = df_show.sort_values(sort_cols)

    # Mostrar tabla
    st.dataframe(
        df_show,
        use_container_width=True,
    )

    # Botón para descargar CSV
    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar resultado en CSV",
        data=csv,
        file_name="partidos_clasificados.csv",
        mime="text/csv",
    )

    # Resumen rápido
    if not picks.empty:
        n_ideal = (picks["PickType"] == "Ideal").sum()
        n_buena_f = (picks["PickType"] == "Buena filtrada").sum()
        st.markdown(
            f"**Resumen picks:** {len(picks)} partidos → "
            f"{n_ideal} Ideal, {n_buena_f} Buena filtrada."
        )


if __name__ == "__main__":
    main()
