# app/main.py

import os
import sys
from datetime import datetime, date

import pandas as pd
import streamlit as st
import requests

# -------------------------------------------------------------------
# AÑADIR RAÍZ DEL PROYECTO AL PYTHONPATH PARA IMPORTAR backend.model
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
# CONFIGURACIÓN BÁSICA DE LA APP
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Selector de partidos HT/FT",
    layout="wide",
)

# -------------------------------------------------------------------
# CARGA DE HISTÓRICO (CACHEADO)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def _load_hist_and_stats():
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


# -------------------------------------------------------------------
# CARGA DE TABLA DE SEGUIMIENTO DESDE SUPABASE (si está configurado)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def _load_tracking_from_supabase():
    """
    Intenta leer la tabla 'seguimiento' de Supabase usando:
      - SUPABASE_URL
      - SUPABASE_SERVICE_ROLE_KEY o SUPABASE_ANON_KEY

    Devuelve (df, error_msg)
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        return None, "Variables de entorno de Supabase no configuradas (SUPABASE_URL / SUPABASE_*_KEY)."

    table = "seguimiento"
    # REST estándar de Supabase
    endpoint = url.rstrip("/") + f"/rest/v1/{table}?select=*"

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    resp = requests.get(endpoint, headers=headers)
    if not resp.ok:
        return None, f"Error {resp.status_code} al leer Supabase: {resp.text}"

    data = resp.json()
    if not data:
        return pd.DataFrame(), None

    df = pd.DataFrame(data)

    # Normalizar nombres de columnas posibles de fecha
    for col in ["fecha", "Fecha", "date", "Date"]:
        if col in df.columns:
            df["Fecha"] = pd.to_datetime(df[col]).dt.date
            break

    # Construir ROI_pct si se puede
    if "ROI_pct" in df.columns:
        df["ROI_pct"] = pd.to_numeric(df["ROI_pct"], errors="coerce")
    else:
        roi_col = None
        for candidate in ["ROI", "roi"]:
            if candidate in df.columns:
                roi_col = candidate
                break
        if roi_col is not None:
            df["ROI_pct"] = pd.to_numeric(df[roi_col], errors="coerce") * 100.0
        else:
            # Intento con beneficio / stake_total si existen
            stake_col = None
            profit_col = None
            for c in df.columns:
                if c.lower() in ("stake_total", "stake", "importe_total"):
                    stake_col = c
                if c.lower() in ("beneficio", "profit_euros", "profit"):
                    profit_col = c
            if stake_col and profit_col:
                stake = pd.to_numeric(df[stake_col], errors="coerce")
                profit = pd.to_numeric(df[profit_col], errors="coerce")
                df["ROI_pct"] = (profit / stake.replace(0, pd.NA)) * 100.0
            else:
                df["ROI_pct"] = pd.NA

    return df, None


# -------------------------------------------------------------------
# PÁGINA 1: CLASIFICAR FIXTURES
# -------------------------------------------------------------------
def page_clasificar():
    st.title("Clasificación de partidos (modelo HT/FT)")

    with st.spinner("Cargando histórico y estadísticas..."):
        hist, team_stats, div_stats = _load_hist_and_stats()

    st.subheader("1. Subir fixture de Football-Data")
    st.markdown(
        "Sube el fichero **`fixtures.xlsx`** tal cual lo descargas de "
        "[football-data.co.uk](https://www.football-data.co.uk/)."
    )

    uploaded = st.file_uploader(
        "Selecciona el fichero de fixtures",
        type=["xlsx", "xls"],
        key="fixture_uploader",
    )

    if uploaded is None:
        st.info("Sube un fixture para continuar.")
        return

    try:
        fixtures_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error leyendo el Excel: {e}")
        return

    # Aplica modelo
    try:
        scored = score_fixtures(hist, team_stats, div_stats, fixtures_df)
    except Exception as e:
        st.error(f"Error aplicando el modelo a los fixtures: {e}")
        return

    st.subheader("2. Resultado de la clasificación")

    # Selección de columnas a mostrar: info partido + cuotas + scores + clasificación
    preferred_cols = [
        "Date", "Time", "Div",
        "HomeTeam", "AwayTeam",
        "B365H", "B365D", "B365A",
        "L_score", "LeagueTier",
        "H_T_score", "A_T_score",
        "MatchScore", "MatchClass", "PickType",
    ]
    cols_to_show = [c for c in preferred_cols if c in scored.columns]
    if not cols_to_show:
        # Si por lo que sea no están, enseño todo
        cols_to_show = scored.columns.tolist()

    # Pequeños filtros en la propia tabla
    left, right = st.columns(2)
    with left:
        # Filtro por clase del modelo
        clases = sorted(list(scored["MatchClass"].dropna().unique()))
        clase_sel = st.multiselect(
            "Filtrar por clasificación del modelo",
            options=clases,
            default=clases,
        )
    with right:
        # Filtro por PickType si existe
        if "PickType" in scored.columns:
            picks = sorted(list(scored["PickType"].dropna().unique()))
            pick_sel = st.multiselect(
                "Filtrar por tipo de pick",
                options=picks,
                default=picks,
            )
        else:
            pick_sel = None

    filtered = scored.copy()
    if clase_sel:
        filtered = filtered[filtered["MatchClass"].isin(clase_sel)]
    if pick_sel and "PickType" in filtered.columns:
        filtered = filtered[filtered["PickType"].isin(pick_sel)]

    # Orden básico por fecha/hora si existen las columnas
    for c in ["Date", "Time"]:
        if c in filtered.columns:
            # Forzamos formato legible
            if c == "Date":
                filtered[c] = pd.to_datetime(filtered[c], errors="coerce").dt.date

    if "Date" in filtered.columns:
        filtered = filtered.sort_values(["Date", "Time"] if "Time" in filtered.columns else ["Date"])

    st.dataframe(filtered[cols_to_show], use_container_width=True)

    st.caption("Solo se muestran las columnas relevantes para la decisión de apuesta.")


# -------------------------------------------------------------------
# PÁGINA 2: SEGUIMIENTO + GRÁFICOS ROI
# -------------------------------------------------------------------
def page_seguimiento():
    st.title("Seguimiento de estrategia y ROI")

    df, err = _load_tracking_from_supabase()
    if err:
        st.error(err)
        st.info("Configura las variables de entorno de Supabase en Render para activar esta sección.")
        return

    if df is None or df.empty:
        st.warning("La tabla 'seguimiento' está vacía o no se han encontrado registros.")
        return

    # Normalizar algunos nombres típicos de columnas
    # Equipos
    home_col = None
    away_col = None
    for c in df.columns:
        l = c.lower()
        if l in ("home", "hometeam", "equipo_local", "local"):
            home_col = c
        if l in ("away", "awayteam", "equipo_visitante", "visitante"):
            away_col = c

    # Fecha
    if "Fecha" not in df.columns:
        for c in df.columns:
            if c.lower() in ("fecha", "date"):
                df["Fecha"] = pd.to_datetime(df[c]).dt.date
                break

    # ROI_pct ya calculado en _load_tracking_from_supabase; podemos limpiar NaN
    df["ROI_pct"] = pd.to_numeric(df["ROI_pct"], errors="coerce")

    # Filtros
    st.subheader("Filtros")

    c1, c2, c3 = st.columns(3)

    with c1:
        if "Fecha" in df.columns:
            min_d = df["Fecha"].min()
            max_d = df["Fecha"].max()
            rango = st.date_input(
                "Rango de fechas",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )
        else:
            rango = None

    with c2:
        equipos = set()
        if home_col:
            equipos.update(df[home_col].dropna().unique())
        if away_col:
            equipos.update(df[away_col].dropna().unique())
        equipos = sorted(list(equipos))
        if equipos:
            eq_sel = st.multiselect("Equipos (local o visitante)", options=equipos)
        else:
            eq_sel = []

    with c3:
        # Filtro por clasificación del modelo si existe
        class_col = None
        for c in df.columns:
            if c.lower() in ("modelclass", "clasificacion_modelo", "model_class"):
                class_col = c
                break
        if class_col:
            clases = sorted(list(df[class_col].dropna().unique()))
            class_sel = st.multiselect("Clasificación modelo", options=clases, default=clases)
        else:
            class_sel = None

    # Aplicar filtros
    fdf = df.copy()

    if rango and "Fecha" in fdf.columns:
        d1, d2 = rango
        fdf = fdf[(fdf["Fecha"] >= d1) & (fdf["Fecha"] <= d2)]

    if eq_sel and (home_col or away_col):
        mask = pd.Series(False, index=fdf.index)
        if home_col:
            mask = mask | fdf[home_col].isin(eq_sel)
        if away_col:
            mask = mask | fdf[away_col].isin(eq_sel)
        fdf = fdf[mask]

    if class_sel and class_col:
        fdf = fdf[fdf[class_col].isin(class_sel)]

    st.subheader("Tabla de seguimiento (con ROI %)")

    # Mostrar columnas clave
    cols_show = []
    for c in ["Fecha", home_col, away_col, "ROI_pct"]:
        if c and c in fdf.columns:
            cols_show.append(c)

    # Añadimos beneficio / stake si existen
    for c in fdf.columns:
        if c.lower() in ("beneficio", "profit_euros", "profit", "stake_total", "stake", "importe_total"):
            if c not in cols_show:
                cols_show.append(c)

    if cols_show:
        st.dataframe(fdf[cols_show], use_container_width=True)
    else:
        st.dataframe(fdf, use_container_width=True)

    st.subheader("Evolución de ROI (%)")

    if "Fecha" in fdf.columns and fdf["ROI_pct"].notna().any():
        # Ordenar por fecha
        fdf = fdf.sort_values("Fecha")
        # ROI medio diario
        daily = fdf.groupby("Fecha")["ROI_pct"].mean().reset_index()

        st.line_chart(
            data=daily.set_index("Fecha")["ROI_pct"],
        )

        st.caption("Línea: ROI medio diario (en %).")
    else:
        st.info("No hay datos suficientes de ROI_pct para graficar.")


# -------------------------------------------------------------------
# ENRUTADOR PRINCIPAL
# -------------------------------------------------------------------
def main():
    menu = ["Clasificar partidos", "Seguimiento"]
    choice = st.sidebar.selectbox("Menú", menu)

    if choice == "Clasificar partidos":
        page_clasificar()
    else:
        page_seguimiento()


if __name__ == "__main__":
    main()
