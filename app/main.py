# app/main.py

import os
import sys
import io
import traceback

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
    update_seguimiento_row,
)

from backend.banca import (  # type: ignore
    fetch_banca_movimientos,
    insert_banca_movimiento,
)

# ✅ VIX (vix_daily + órdenes)
from backend.vix import (  # type: ignore
    run_vix_pipeline,
    fetch_vix_daily,
    fetch_vix_orders,
    insert_vix_order,
    update_vix_order_status,
)

# ✅ Motor de trading VIX (posiciones) — import seguro (no rompe la app si falta)
VIX_TRADING_AVAILABLE = True
VIX_TRADING_IMPORT_ERROR = None
try:
    from backend.vix_trading import (  # type: ignore
        run_vix_execution,
        fetch_vix_positions,
    )
except Exception as e:
    VIX_TRADING_AVAILABLE = False
    run_vix_execution = None  # type: ignore
    fetch_vix_positions = None  # type: ignore
    VIX_TRADING_IMPORT_ERROR = "".join(traceback.format_exception(type(e), e, e.__traceback__))


# ---------- CARGA HISTÓRICO (CACHEADO) ----------
@st.cache_data(show_spinner="Cargando histórico y calculando estadísticas…")
def _load_hist_and_stats():
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


@st.cache_data(show_spinner=False, ttl=60)
def _fetch_vix_daily_cached():
    # cache cortito para no machacar Supabase con cada rerun
    return fetch_vix_daily()


# ======================================================================
# HELPERS (ROI / RAROC / BANCA)
# ======================================================================

def _safe_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _safe_int_default(value, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _compute_total_stake(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("stake_btts_no", 0).fillna(0)
        + df.get("stake_u35", 0).fillna(0)
        + df.get("stake_1_1", 0).fillna(0)
    )


def _compute_roi_calc(df: pd.DataFrame) -> pd.Series:
    total_stake = _compute_total_stake(df)
    roi = pd.Series(pd.NA, index=df.index, dtype="object")
    profit = df.get("profit_euros", pd.Series(index=df.index))
    ok = (total_stake > 0) & profit.notna()
    roi.loc[ok] = profit.loc[ok] / total_stake.loc[ok]
    return roi


def _compute_raroc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "roi_calc" not in out.columns:
        out["roi_calc"] = _compute_roi_calc(out)

    _safe_numeric(out, "close_minute_global")
    out["raroc"] = pd.NA
    out["raroc_pct"] = pd.NA

    ok = out["roi_calc"].notna() & out["close_minute_global"].notna() & (out["close_minute_global"] > 0)
    out.loc[ok, "raroc"] = pd.to_numeric(out.loc[ok, "roi_calc"], errors="coerce") / out.loc[ok, "close_minute_global"]
    out.loc[ok, "raroc_pct"] = pd.to_numeric(out.loc[ok, "raroc"], errors="coerce") * 100.0
    return out


def _banca_sign(tipo: str, importe: float) -> float:
    t = (tipo or "").upper().strip()
    if t == "DEPOSITO":
        return abs(importe)
    if t == "RETIRADA":
        return -abs(importe)
    return float(importe)


def _compute_equity_curve(
    movimientos_df: pd.DataFrame,
    seguimiento_df: pd.DataFrame,
    only_real: bool = True,
) -> pd.DataFrame:
    mov = movimientos_df.copy()
    if mov.empty:
        mov = pd.DataFrame(columns=["fecha", "tipo", "importe"])

    if "fecha" in mov.columns:
        mov["fecha"] = pd.to_datetime(mov["fecha"], errors="coerce").dt.date
    _safe_numeric(mov, "importe")

    if "tipo" in mov.columns:
        mov["signed"] = mov.apply(lambda r: _banca_sign(r.get("tipo", ""), r.get("importe", 0.0) or 0.0), axis=1)
    else:
        mov["signed"] = mov.get("importe", 0.0)

    mov_daily = (
        mov.dropna(subset=["fecha"])
        .groupby("fecha", as_index=False)["signed"]
        .sum()
        .rename(columns={"signed": "movimientos"})
    )

    seg = seguimiento_df.copy()
    if seg.empty:
        seg = pd.DataFrame(columns=["fecha", "profit_euros", "apuesta_real"])

    if "fecha" in seg.columns:
        seg["fecha"] = pd.to_datetime(seg["fecha"], errors="coerce").dt.date
    _safe_numeric(seg, "profit_euros")

    if only_real and "apuesta_real" in seg.columns:
        seg = seg[seg["apuesta_real"] == "SI"].copy()

    profit_daily = (
        seg.dropna(subset=["fecha"])
        .groupby("fecha", as_index=False)["profit_euros"]
        .sum()
        .rename(columns={"profit_euros": "profit"})
    )

    fechas = pd.Series(pd.concat([
        mov_daily.get("fecha", pd.Series(dtype="object")),
        profit_daily.get("fecha", pd.Series(dtype="object")),
    ], ignore_index=True)).dropna().unique()

    if len(fechas) == 0:
        return pd.DataFrame(columns=["fecha", "movimientos", "profit", "equity"])

    cal = pd.DataFrame({"fecha": sorted(fechas)})
    cal = cal.merge(mov_daily, on="fecha", how="left").merge(profit_daily, on="fecha", how="left")
    cal["movimientos"] = cal["movimientos"].fillna(0.0)
    cal["profit"] = cal["profit"].fillna(0.0)

    cal["mov_cum"] = cal["movimientos"].cumsum()
    cal["profit_cum"] = cal["profit"].cumsum()
    cal["equity"] = cal["mov_cum"] + cal["profit_cum"]

    cal["peak"] = cal["equity"].cummax()
    cal["drawdown_abs"] = cal["equity"] - cal["peak"]
    cal["drawdown_pct"] = pd.NA
    ok = cal["peak"] > 0
    cal.loc[ok, "drawdown_pct"] = cal.loc[ok, "drawdown_abs"] / cal.loc[ok, "peak"]

    return cal


def _excel_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel NO soporta datetimes con timezone.
    Convierte cualquier datetime tz-aware a naive (sin tz).
    """
    out = df.copy()

    # Columnas datetime tz-aware
    try:
        tz_cols = out.select_dtypes(include=["datetimetz"]).columns.tolist()
        for c in tz_cols:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.tz_localize(None)
    except Exception:
        pass

    # Columnas object que podrían contener Timestamp con tz
    for c in out.columns:
        if out[c].dtype == "object":
            # intentamos detectar si hay timestamps tz-aware
            sample = out[c].dropna().head(20).tolist()
            has_tz = False
            for v in sample:
                try:
                    if isinstance(v, pd.Timestamp) and v.tzinfo is not None:
                        has_tz = True
                        break
                except Exception:
                    pass
            if has_tz:
                out[c] = pd.to_datetime(out[c], errors="coerce")
                try:
                    out[c] = out[c].dt.tz_localize(None)
                except Exception:
                    # si viene tz-aware real
                    try:
                        out[c] = out[c].dt.tz_convert(None)
                    except Exception:
                        pass

    return out


# ======================================================================
# VISTA: SELECTOR DE PARTIDOS
# ======================================================================
def show_selector():
    st.markdown("### 1. Cargar fixture (Football-Data)")

    uploaded = st.file_uploader(
        "Sube el fichero `fixtures.xlsx` (tal cual de football-data.co.uk)",
        type=["xlsx", "xls"],
    )

    if uploaded is None:
        st.info("Sube un fixture para continuar.")
        return

    try:
        fixtures_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error leyendo el fichero de fixtures: {e}")
        return

    expected_cols = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    missing = [c for c in expected_cols if c not in fixtures_raw.columns]
    if missing:
        st.error(f"Faltan columnas en el fixture: {missing}")
        return

    fixtures_df = fixtures_raw[expected_cols].copy()

    try:
        _, team_stats, div_stats = _load_hist_and_stats()
    except Exception as e:
        st.error(f"Error cargando histórico/modelo: {e}")
        return

    try:
        scored = score_fixtures(team_stats, div_stats, fixtures_df)
    except Exception as e:
        st.error(f"Error aplicando el modelo a los fixtures: {e}")
        return

    picks = scored[
        scored["MatchClass"].isin(["Ideal", "Buena", "Buena filtrada"])
        | scored["PickType"].notna()
    ].copy()

    st.markdown("### 2. Resultados del modelo")

    if picks.empty:
        st.warning("Ningún partido cumple los filtros de visualización para este fixture.")
        return

    cols_to_show = [
        "Date", "Time", "Div", "HomeTeam", "AwayTeam",
        "B365H", "B365D", "B365A",
        "L_score", "LeagueTier", "H_T_score", "A_T_score",
        "MatchScore", "MatchClass", "PickType",
    ]

    base_table = picks[cols_to_show].sort_values(["Date", "Time", "Div", "HomeTeam"]).copy()
    base_table["Seleccionar"] = False

    st.markdown("#### Selecciona los partidos a los que vas a apostar (o a guardar para seguimiento)")

    edited = st.data_editor(
        base_table,
        use_container_width=True,
        key="tabla_picks_con_seleccion",
        column_config={
            "Seleccionar": st.column_config.CheckboxColumn(
                "Seleccionar",
                help="Marca los partidos que quieres guardar en seguimiento",
                default=False,
            )
        },
    )

    seleccionados = edited[edited["Seleccionar"] == True].copy()

    st.markdown("### 3. Guardar selección en Supabase")

    col1, _ = st.columns([1, 3])
    with col1:
        st.write(f"Partidos seleccionados: **{len(seleccionados)}**")

    guardar = st.button("Guardar seleccionados en Supabase")

    if guardar:
        if seleccionados.empty:
            st.warning("No has seleccionado ningún partido.")
        else:
            try:
                seleccionados_sin_flag = seleccionados.drop(columns=["Seleccionar"])
                merge_cols = ["Date", "Time", "Div", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]

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


# ======================================================================
# VISTA: GESTIÓN DE APUESTAS
# ======================================================================
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

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    with st.expander("Filtros"):
        if "fecha" in df.columns and df["fecha"].notna().any():
            min_date = df["fecha"].min().date()
            max_date = df["fecha"].max().date()
            fecha_desde, fecha_hasta = st.date_input("Rango de fechas", value=(min_date, max_date))
        else:
            fecha_desde, fecha_hasta = None, None

        pick_types = sorted([x for x in df.get("pick_type", pd.Series()).dropna().unique()])
        pick_filter = st.multiselect("Filtrar por PickType", options=pick_types, default=pick_types) if pick_types else []

        divisiones = sorted([x for x in df.get("division", pd.Series()).dropna().unique()])
        div_filter = st.multiselect("Filtrar por división", options=divisiones, default=divisiones) if divisiones else []

        equipos = sorted(set(df.get("home_team", pd.Series()).dropna().unique()) | set(df.get("away_team", pd.Series()).dropna().unique()))
        equipos_filter = st.multiselect("Filtrar por equipo (local o visitante)", options=equipos, default=[]) if equipos else []

        if "apuesta_real" in df.columns:
            ar_values = sorted([x for x in df["apuesta_real"].dropna().unique()])
            apuesta_real_filter = st.multiselect("Filtrar por apuesta_real (SI/NO)", options=ar_values, default=ar_values) if ar_values else []
        else:
            apuesta_real_filter = []

        if "estrategia" in df.columns:
            est_values = sorted([x for x in df["estrategia"].dropna().unique()])
            estrategia_filter = st.multiselect("Filtrar por estrategia", options=est_values, default=est_values) if est_values else []
        else:
            estrategia_filter = []

    mask = pd.Series(True, index=df.index)
    if fecha_desde is not None and "fecha" in df.columns:
        mask &= df["fecha"].dt.date >= fecha_desde
    if fecha_hasta is not None and "fecha" in df.columns:
        mask &= df["fecha"].dt.date <= fecha_hasta
    if pick_filter:
        mask &= df["pick_type"].isin(pick_filter)
    if div_filter:
        mask &= df["division"].isin(div_filter)
    if equipos_filter:
        mask &= df["home_team"].isin(equipos_filter) | df["away_team"].isin(equipos_filter)
    if apuesta_real_filter and "apuesta_real" in df.columns:
        mask &= df["apuesta_real"].isin(apuesta_real_filter)
    if estrategia_filter and "estrategia" in df.columns:
        mask &= df["estrategia"].isin(estrategia_filter)

    filtered = df[mask].copy()
    if filtered.empty:
        st.warning("No hay registros que cumplan los filtros.")
        return

    st.write(f"Registros filtrados: **{len(filtered)}**")

    base_editable = [
        "stake_btts_no", "stake_u35", "stake_1_1",
        "close_minute_global", "close_minute_1_1",
        "odds_btts_no_init", "odds_u35_init", "odds_1_1_init",
        "profit_euros", "roi", "apuesta_real",
        "minuto_primer_gol", "pct_minuto_primer_gol",
        "estrategia",
        "raroc", "raroc_pct",
    ]
    editable_cols = [c for c in base_editable if c in filtered.columns]

    if "fecha" in filtered.columns:
        filtered = filtered.sort_values(["fecha", "hora", "division", "home_team"], na_position="last")

    st.markdown("#### Edición rápida (tabla)")
    edited = st.data_editor(filtered, use_container_width=True, key="editor_seguimiento", hide_index=True)

    if st.button("Guardar cambios en Supabase (tabla)"):
        try:
            updated = update_seguimiento_from_df(original_df=filtered, edited_df=edited, editable_cols=editable_cols)
            st.success(f"Se han actualizado {updated} filas en la tabla 'seguimiento'.")
        except Exception as e:
            st.error(f"Error actualizando en Supabase: {e}")

    st.markdown("---")
    st.markdown("#### Edición detallada (modo formulario)")

    if "id" not in filtered.columns:
        st.warning("No hay columna 'id' en los datos, no se puede usar el modo formulario.")
        return

    opciones = []
    for _, row in filtered.iterrows():
        etiqueta = (
            f"ID {row['id']} - {row.get('fecha', '')} - {row.get('division', '')} - "
            f"{row.get('home_team', '')} vs {row.get('away_team', '')}"
        )
        opciones.append((int(row["id"]), etiqueta))

    ids = [o[0] for o in opciones]
    labels = [o[1] for o in opciones]

    seleccion = st.selectbox(
        "Selecciona una apuesta para editar en detalle",
        options=list(range(len(ids))),
        format_func=lambda i: labels[i],
    )

    selected_id = ids[seleccion]
    row_sel = filtered[filtered["id"] == selected_id].iloc[0]

    with st.form("form_edicion_detallada"):
        st.write(
            f"**Partido:** {row_sel.get('home_team', '')} vs {row_sel.get('away_team', '')} "
            f"({row_sel.get('division', '')}, {row_sel.get('fecha', '')}, {row_sel.get('hora', '')})"
        )

        estrategia = None
        if "estrategia" in filtered.columns:
            estrategia_actual = (row_sel.get("estrategia") or "Convexidad")
            estrategia = st.selectbox(
                "Estrategia",
                options=["Convexidad", "Spread Attack"],
                index=0 if estrategia_actual == "Convexidad" else 1,
            )

        stake_btts_no = st.number_input("Stake BTTS NO", value=float(row_sel.get("stake_btts_no", 0) or 0), step=1.0)
        stake_u35 = st.number_input("Stake Under 3.5", value=float(row_sel.get("stake_u35", 0) or 0), step=1.0)
        stake_1_1 = st.number_input("Stake marcador 1-1", value=float(row_sel.get("stake_1_1", 0) or 0), step=1.0)

        close_minute_global = st.number_input(
            "Minuto de cierre global",
            value=_safe_int_default(row_sel.get("close_minute_global"), 0),
            step=1,
        )
        close_minute_1_1 = st.number_input(
            "Minuto de cierre 1-1",
            value=_safe_int_default(row_sel.get("close_minute_1_1"), 0),
            step=1,
        )

        odds_btts_no_init = st.number_input("Cuota inicial BTTS NO", value=float(row_sel.get("odds_btts_no_init", 0) or 0), step=0.01)
        odds_u35_init = st.number_input("Cuota inicial Under 3.5", value=float(row_sel.get("odds_u35_init", 0) or 0), step=0.01)
        odds_1_1_init = st.number_input("Cuota inicial 1-1", value=float(row_sel.get("odds_1_1_init", 0) or 0), step=0.01)

        profit_euros = st.number_input("Profit (€)", value=float(row_sel.get("profit_euros", 0) or 0), step=1.0)

        minuto_primer_gol_actual = row_sel.get("minuto_primer_gol")
        if pd.isna(minuto_primer_gol_actual):
            minuto_primer_gol_actual = 0
        minuto_primer_gol = st.number_input(
            "Minuto del primer gol (si no hubo, marca la casilla para guardar NULL)",
            value=int(minuto_primer_gol_actual),
            min_value=0,
            max_value=130,
            step=1,
        )
        sin_gol = st.checkbox("No hubo gol (guardar NULL)", value=pd.isna(row_sel.get("minuto_primer_gol")))

        total_stake = stake_btts_no + stake_u35 + stake_1_1
        if total_stake > 0:
            roi_calc = profit_euros / total_stake
            st.write(f"ROI calculado: **{roi_calc*100:.2f}%** (profit / suma de stakes)")
        else:
            roi_calc = None
            st.write("ROI calculado: — (faltan stakes o profit)")

        raroc_calc = None
        raroc_pct_calc = None
        if roi_calc is not None and close_minute_global and close_minute_global > 0:
            raroc_calc = float(roi_calc) / float(close_minute_global)
            raroc_pct_calc = raroc_calc * 100.0
            st.write(f"RAROC: **{raroc_pct_calc:.4f}% por minuto**")
        else:
            st.write("RAROC: — (necesita ROI y close_minute_global > 0).")

        apuesta_real_actual = row_sel.get("apuesta_real") or "NO"
        apuesta_real = st.selectbox("¿Apuesta real?", options=["SI", "NO"], index=0 if apuesta_real_actual == "SI" else 1)

        submitted = st.form_submit_button("Guardar cambios (formulario)")

        if submitted:
            cambios = {
                "stake_btts_no": stake_btts_no,
                "stake_u35": stake_u35,
                "stake_1_1": stake_1_1,
                "close_minute_global": close_minute_global,
                "close_minute_1_1": close_minute_1_1,
                "odds_btts_no_init": odds_btts_no_init,
                "odds_u35_init": odds_u35_init,
                "odds_1_1_init": odds_1_1_init,
                "profit_euros": profit_euros,
                "apuesta_real": apuesta_real,
                "minuto_primer_gol": None if sin_gol else int(minuto_primer_gol),
                "roi": roi_calc if roi_calc is not None else None,
            }

            if estrategia is not None and "estrategia" in filtered.columns:
                cambios["estrategia"] = estrategia

            if "raroc" in filtered.columns:
                cambios["raroc"] = raroc_calc
            if "raroc_pct" in filtered.columns:
                cambios["raroc_pct"] = raroc_pct_calc

            try:
                update_seguimiento_row(selected_id, cambios)
                st.success(f"Registro ID {selected_id} actualizado correctamente.")
            except Exception as e:
                st.error(f"Error actualizando (formulario) en Supabase: {e}")


# ======================================================================
# VISTA: ESTADÍSTICAS ROI (+ VIX)
# ======================================================================
def show_stats():
    st.markdown("### Estadísticas de ROI")

    try:
        df = fetch_seguimiento()
    except Exception as e:
        st.error(f"Error cargando datos de seguimiento: {e}")
        return

    if df.empty:
        st.info("Todavía no hay apuestas en la tabla 'seguimiento'.")
        return

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    for col in ["stake_btts_no", "stake_u35", "stake_1_1", "profit_euros"]:
        _safe_numeric(df, col)

    df["total_stake"] = _compute_total_stake(df)
    df["roi_calc"] = _compute_roi_calc(df)

    apuesta_real_opts = sorted([x for x in df.get("apuesta_real", pd.Series()).dropna().unique()])
    if apuesta_real_opts:
        ar_sel = st.multiselect("Filtrar por tipo de apuesta", options=apuesta_real_opts, default=apuesta_real_opts)
        df = df[df["apuesta_real"].isin(ar_sel)]

    if df.empty:
        st.warning("No hay registros tras aplicar filtros.")
        return

    total_profit = df["profit_euros"].fillna(0).sum() if "profit_euros" in df.columns else 0.0
    total_stake_sum = df["total_stake"].fillna(0).sum()
    roi_global = (total_profit / total_stake_sum) if total_stake_sum > 0 else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total profit (€)", f"{total_profit:,.2f}")
    with col
