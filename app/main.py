# app/main.py

import os
import sys

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
from backend.seguimiento import (  # type: ignore
    insert_seguimiento_from_picks,
    fetch_seguimiento,
    update_seguimiento_from_df,
)


# ---------- CARGA HISTÓRICO (CACHEADO) ----------

@st.cache_data(show_spinner="Cargando histórico y calculando estadísticas…")
def _load_hist_and_stats():
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


# ---------- VISTA: SELECTOR DE PARTIDOS ----------

def show_selector():
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

    # Cargar histórico + stats para el modelo
    try:
        _, team_stats, div_stats = _load_hist_and_stats()
    except Exception as e:
        st.error(f"Error cargando histórico/modelo: {e}")
        return

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

    # Tabla reducida
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

    base_table = (
        picks[cols_to_show]
        .sort_values(["Date", "Time", "Div", "HomeTeam"])
        .copy()
    )

    # Añadimos columna de selección
    base_table["Seleccionar"] = False

    st.markdown("#### Selecciona los partidos a los que vas a apostar")

    edited = st.data_editor(
        base_table,
        use_container_width=True,
        key="tabla_picks_con_seleccion",
        column_config={
            "Seleccionar": st.column_config.CheckboxColumn(
                "Seleccionar",
                help="Marca los partidos a los que realmente vas a apostar",
                default=False,
            )
        },
    )

    # Filtrar los seleccionados
    seleccionados = edited[edited["Seleccionar"] == True].copy()

    st.markdown("### 3. Guardar selección en Supabase")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.write(f"Partidos seleccionados: **{len(seleccionados)}**")

    guardar = st.button("Guardar seleccionados en Supabase")

    if guardar:
        if seleccionados.empty:
            st.warning("No has seleccionado ningún partido.")
        else:
            try:
                # Eliminamos la columna Seleccionar antes de mergear
                seleccionados_sin_flag = seleccionados.drop(columns=["Seleccionar"])

                merge_cols = [
                    "Date",
                    "Time",
                    "Div",
                    "HomeTeam",
                    "AwayTeam",
                    "B365H",
                    "B365D",
                    "B365A",
                ]

                # Volvemos a unir con 'picks' para incluir L_score, H_T_score, etc.
                seleccionados_full = seleccionados_sin_flag.merge(
                    picks,
                    on=merge_cols,
                    how="left",
                    suffixes=("", "_y"),
                )

                insert_seguimiento_from_picks(seleccionados_full)

                st.success("Partidos guardados en la tabla 'seguimiento' de Supabase.")
            except Exception as e:
                st.error(f"Error guardando en Supabase: {e}")


# ---------- VISTA: GESTIÓN DE APUESTAS ----------

def show_gestion():
    st.markdown("### Gestión de apuestas guardadas (`seguimiento`)")

    try:
        df = fetch_seguimiento()
    except Exception as e:
        st.error(f"Error cargando datos de seguimiento: {e}")
        return

    if df.empty:
        st.info("Todavía no hay apuestas guardadas en la tabla 'seguimiento'.")
        return

    # Normalizamos fecha para filtrar
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Filtros
    with st.expander("Filtros"):
        # Rango de fechas
        if "fecha" in df.columns and df["fecha"].notna().any():
            min_date = df["fecha"].min().date()
            max_date = df["fecha"].max().date()
            fecha_desde, fecha_hasta = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
            )
        else:
            fecha_desde, fecha_hasta = None, None

        # Filtro por pick_type
        pick_types = sorted([x for x in df.get("pick_type", pd.Series()).dropna().unique()])
        if pick_types:
            pick_filter = st.multiselect(
                "Filtrar por PickType",
                options=pick_types,
                default=pick_types,
            )
        else:
            pick_filter = []

        # Filtro por division
        divisiones = sorted([x for x in df.get("division", pd.Series()).dropna().unique()])
        if divisiones:
            div_filter = st.multiselect(
                "Filtrar por división",
                options=divisiones,
                default=divisiones,
            )
        else:
            div_filter = []

    # Aplicar filtros
    mask = pd.Series(True, index=df.index)

    if fecha_desde is not None and "fecha" in df.columns:
        mask &= df["fecha"].dt.date >= fecha_desde
    if fecha_hasta is not None and "fecha" in df.columns:
        mask &= df["fecha"].dt.date <= fecha_hasta

    if pick_filter:
        mask &= df["pick_type"].isin(pick_filter)
    if div_filter:
        mask &= df["division"].isin(div_filter)

    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No hay registros que cumplan los filtros.")
        return

    st.write(f"Registros filtrados: **{len(filtered)}**")

    # Columnas editables
    editable_cols = [
        "stake_btts_no",
        "stake_u35",
        "stake_1_1",
        "close_minute_global",
        "close_minute_1_1",
        "odds_btts_no_init",
        "odds_u35_init",
        "odds_1_1_init",
        "profit_euros",
        "roi",
    ]

    # Ordenamos por fecha/hora para que sea más legible
    if "fecha" in filtered.columns:
        filtered = filtered.sort_values(["fecha", "hora", "division", "home_team"])

    # Mostramos editor
    edited = st.data_editor(
        filtered,
        use_container_width=True,
        key="editor_seguimiento",
        hide_index=True,
    )

    if st.button("Guardar cambios en Supabase"):
        try:
            updated = update_seguimiento_from_df(
                original_df=filtered,
                edited_df=edited,
                editable_cols=editable_cols,
            )
            st.success(f"Se han actualizado {updated} filas en la tabla 'seguimiento'.")
        except Exception as e:
            st.error(f"Error actualizando en Supabase: {e}")


# ---------- INTERFAZ PRINCIPAL ----------

def main():
    st.set_page_config(
        page_title="Selector de Partidos HT/FT",
        layout="wide",
    )

    st.sidebar.title("Navegación")
    modo = st.sidebar.radio(
        "Selecciona modo",
        options=["Selector de partidos", "Gestión de apuestas"],
    )

    if modo == "Selector de partidos":
        st.title("Selector de partidos HT/FT – Modelo")
        show_selector()
    else:
        st.title("Gestión de apuestas – Seguimiento")
        show_gestion()


if __name__ == "__main__":
    main()
