# app/main.py
# ============================================================
# APP PRINCIPAL – HT/FT + VIX + MOTOR DE POSICIONES
# ============================================================

import os
import sys
import io
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PATH
# ------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ------------------------------------------------------------
# IMPORTS CORE
# ------------------------------------------------------------
from backend.model import (
    load_historical_data,
    compute_team_and_league_stats,
    score_fixtures,
)

from backend.seguimiento import (
    insert_seguimiento_from_picks,
    fetch_seguimiento,
    update_seguimiento_from_df,
    update_seguimiento_row,
)

from backend.banca import (
    fetch_banca_movimientos,
    insert_banca_movimiento,
)

# -------------------- VIX DAILY + ÓRDENES --------------------
from backend.vix import (
    run_vix_pipeline,
    fetch_vix_daily,
    fetch_vix_orders,
    insert_vix_order,
    update_vix_order_status,
)

# -------------------- MOTOR VIX (POSICIONES) -----------------
VIX_TRADING_AVAILABLE = True
try:
    from backend.vix_trading import run_vix_execution, fetch_vix_positions
except Exception:
    VIX_TRADING_AVAILABLE = False
    run_vix_execution = None
    fetch_vix_positions = None


# ============================================================
# CACHE
# ============================================================
@st.cache_data(show_spinner="Cargando histórico…")
def _load_hist_and_stats():
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


@st.cache_data(ttl=60)
def _fetch_vix_daily_cached():
    return fetch_vix_daily()


# ============================================================
# HELPERS
# ============================================================
def _safe_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quita timezone de TODAS las columnas datetime
    (Excel NO soporta tz-aware)
    """
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            try:
                df[c] = df[c].dt.tz_localize(None)
            except Exception:
                pass
    return df


# ============================================================
# SELECTOR PARTIDOS
# ============================================================
def show_selector():
    st.markdown("### Selector de partidos")

    uploaded = st.file_uploader(
        "Sube `fixtures.xlsx` (Football-Data)",
        type=["xlsx", "xls"],
    )
    if uploaded is None:
        return

    fixtures = pd.read_excel(uploaded)

    needed = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    if not all(c in fixtures.columns for c in needed):
        st.error("Faltan columnas en el fichero.")
        return

    fixtures = fixtures[needed]

    _, team_stats, div_stats = _load_hist_and_stats()
    scored = score_fixtures(team_stats, div_stats, fixtures)

    picks = scored[
        (scored["MatchClass"].isin(["Ideal", "Buena", "Buena filtrada"]))
        | scored["PickType"].notna()
    ].copy()

    picks["Seleccionar"] = False
    edited = st.data_editor(picks, hide_index=True)

    selected = edited[edited["Seleccionar"] == True]
    if st.button("Guardar en seguimiento"):
        insert_seguimiento_from_picks(selected)
        st.success("Guardado.")


# ============================================================
# VIX VIEW
# ============================================================
def show_vix():
    st.title("VIX – Régimen y Motor")

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start", pd.Timestamp("2020-01-01").date())
    with col2:
        end = st.date_input("End", pd.Timestamp.today().date())

    if st.button("Actualizar VIX"):
        run_vix_pipeline(str(start), str(end))
        st.success("VIX actualizado")

    daily = fetch_vix_daily()
    if daily.empty:
        st.info("Sin datos VIX")
        return

    daily["fecha"] = pd.to_datetime(daily["fecha"], errors="coerce")
    daily = daily.sort_values("fecha")

    last = daily.iloc[-1]
    st.metric("Estado", last.get("estado", "—"))
    st.metric("Acción", last.get("accion", "—"))
    st.write(last.get("comentario", ""))

    st.dataframe(daily.tail(200), use_container_width=True)

    # EXPORT
    export_df = _strip_tz(daily.copy())
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        export_df.to_excel(w, index=False)
    st.download_button(
        "Descargar vix_daily.xlsx",
        buf.getvalue(),
        "vix_daily.xlsx",
    )

    # ---------------- MOTOR ----------------
    st.markdown("---")
    st.markdown("## Motor VIX")

    if not VIX_TRADING_AVAILABLE:
        st.error("Motor VIX NO disponible")
        return

    if st.button("▶ Ejecutar motor"):
        run_vix_execution()
        st.success("Motor ejecutado")

    pos = fetch_vix_positions()
    if not pos.empty:
        pos = _strip_tz(pos)
        st.dataframe(pos, use_container_width=True)

        buf2 = io.BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as w:
            pos.to_excel(w, index=False)
        st.download_button(
            "Descargar vix_positions.xlsx",
            buf2.getvalue(),
            "vix_positions.xlsx",
        )


# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Menú")

    option = st.sidebar.radio(
        "Sección",
        ["Selector", "VIX"],
    )

    if option == "Selector":
        show_selector()
    else:
        show_vix()


if __name__ == "__main__":
    main()
