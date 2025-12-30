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
    update_seguimiento_row,
)

from backend.banca import (  # type: ignore
    fetch_banca_movimientos,
    insert_banca_movimiento,
)

# ✅ VIX (solo vix_daily + (opcional) órdenes)
from backend.vix import (  # type: ignore
    run_vix_pipeline,
    fetch_vix_daily,
)

# Intentamos importar órdenes VIX, pero sin romper la app si aún no existen en backend/vix.py
try:
    from backend.vix import (  # type: ignore
        fetch_vix_orders,
        insert_vix_order,
        update_vix_order_status,
    )
except Exception:
    fetch_vix_orders = None
    insert_vix_order = None
    update_vix_order_status = None


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
    with col2:
        st.metric("Total stake (€)", f"{total_stake_sum:,.2f}")
    with col3:
        st.metric("ROI global", f"{roi_global*100:.2f}%" if roi_global is not None else "—")

    st.markdown("---")
    st.markdown("### VIX y ROI (cruce por régimen)")

    try:
        vix_daily = _fetch_vix_daily_cached()
    except Exception as e:
        vix_daily = pd.DataFrame()
        st.warning(f"No se pudo leer vix_daily para cruzar con ROI: {e}")

    if not vix_daily.empty and "fecha" in vix_daily.columns:
        v = vix_daily.copy()
        v["fecha"] = pd.to_datetime(v["fecha"], errors="coerce").dt.date
        v = v.dropna(subset=["fecha"])

        work = df.copy()
        work["fecha_d"] = pd.to_datetime(work["fecha"], errors="coerce").dt.date

        if "estado" in v.columns:
            work = work.merge(
                v[["fecha", "estado"]].rename(columns={"fecha": "fecha_d", "estado": "vix_estado"}),
                on="fecha_d",
                how="left",
            )
        else:
            work["vix_estado"] = pd.NA

        if work["vix_estado"].notna().any():
            grp = (
                work.groupby("vix_estado")
                .apply(lambda g: pd.Series({
                    "profit_total": g["profit_euros"].fillna(0).sum(),
                    "stake_total": g["total_stake"].fillna(0).sum(),
                    "n": len(g),
                }))
                .reset_index()
            )
            grp["roi_pct"] = grp.apply(
                lambda r: (r["profit_total"] / r["stake_total"] * 100.0) if r["stake_total"] > 0 else 0.0,
                axis=1,
            )
            grp = grp.sort_values("roi_pct", ascending=False)
            st.markdown("#### ROI por régimen VIX (solo días con VIX)")
            st.dataframe(grp, use_container_width=True)
        else:
            st.info("No hay cruce con régimen VIX todavía (faltan fechas o 'estado' en vix_daily).")
    else:
        st.info("Todavía no hay datos VIX suficientes para cruzar con ROI.")

    st.markdown("---")
    st.markdown("### Gráficos de ROI (en %)")

    if "fecha" in df.columns and df["fecha"].notna().any():
        df_time = df[df["fecha"].notna()].sort_values("fecha").copy()
        df_time["profit_acum"] = df_time["profit_euros"].fillna(0).cumsum()
        df_time["stake_acum"] = df_time["total_stake"].fillna(0).cumsum()

        df_time["roi_acum_pct"] = pd.NA
        ok = df_time["stake_acum"] > 0
        df_time.loc[ok, "roi_acum_pct"] = (df_time.loc[ok, "profit_acum"] / df_time.loc[ok, "stake_acum"]) * 100.0

        st.markdown("#### ROI acumulado en el tiempo (%)")
        st.line_chart(df_time.set_index("fecha")[["roi_acum_pct"]], use_container_width=True)
    else:
        st.info("No hay fechas válidas para dibujar ROI acumulado.")

    if "division" in df.columns:
        st.markdown("#### ROI por división (barras, %)")
        grp_div = (
            df.groupby("division")
            .apply(lambda g: pd.Series({
                "profit_total": g["profit_euros"].fillna(0).sum(),
                "stake_total": g["total_stake"].fillna(0).sum(),
            }))
            .reset_index()
        )
        grp_div["roi_pct"] = grp_div.apply(
            lambda r: (r["profit_total"] / r["stake_total"] * 100.0) if r["stake_total"] > 0 else 0.0,
            axis=1,
        )
        grp_div = grp_div.sort_values("roi_pct", ascending=False)
        st.bar_chart(grp_div.set_index("division")[["roi_pct"]], use_container_width=True)


# ======================================================================
# VISTA: SUPERVIVENCIA & CONVEXIDAD (recuperada)
# ======================================================================
def show_supervivencia_convexidad():
    st.markdown("### Supervivencia & Convexidad – Minuto del primer gol (HT)")

    try:
        df = fetch_seguimiento()
    except Exception as e:
        st.error(f"Error cargando datos de seguimiento: {e}")
        return

    if df.empty:
        st.info("Todavía no hay datos en 'seguimiento'.")
        return

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    if "minuto_primer_gol" not in df.columns:
        st.warning("La tabla 'seguimiento' no tiene la columna 'minuto_primer_gol'.")
        return

    df["minuto_primer_gol"] = pd.to_numeric(df["minuto_primer_gol"], errors="coerce")

    with st.expander("Filtros"):
        if "fecha" in df.columns and df["fecha"].notna().any():
            min_date = df["fecha"].min().date()
            max_date = df["fecha"].max().date()
            fecha_desde, fecha_hasta = st.date_input("Rango de fechas", value=(min_date, max_date))
        else:
            fecha_desde, fecha_hasta = None, None

        divisiones = sorted([x for x in df.get("division", pd.Series()).dropna().unique()])
        div_filter = st.multiselect("Filtrar por división", options=divisiones, default=divisiones) if divisiones else []

        pick_types = sorted([x for x in df.get("pick_type", pd.Series()).dropna().unique()])
        pick_filter = st.multiselect("Filtrar por PickType", options=pick_types, default=pick_types) if pick_types else []

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
    if div_filter:
        mask &= df["division"].isin(div_filter)
    if pick_filter:
        mask &= df["pick_type"].isin(pick_filter)
    if apuesta_real_filter and "apuesta_real" in df.columns:
        mask &= df["apuesta_real"].isin(apuesta_real_filter)
    if estrategia_filter and "estrategia" in df.columns:
        mask &= df["estrategia"].isin(estrategia_filter)

    df_filt = df[mask].copy()
    if df_filt.empty:
        st.warning("No hay registros que cumplan los filtros.")
        return

    st.write(f"Partidos considerados: **{len(df_filt)}**")

    minutos = df_filt["minuto_primer_gol"].copy()
    sin_gol_mask = minutos.isna()
    total = len(df_filt)

    grid = list(range(0, 46))
    supervivencia = []
    for m in grid:
        vivos = ((minutos > m) | sin_gol_mask).sum()
        supervivencia.append(vivos / total if total > 0 else 0.0)

    surv_df = pd.DataFrame({"minuto": grid, "supervivencia": supervivencia})

    st.markdown("#### Curva de supervivencia (P[sin gol hasta minuto m], 1ª parte)")
    st.line_chart(surv_df.set_index("minuto")[["supervivencia"]], use_container_width=True)

    st.markdown("#### Distribución del minuto del primer gol (bloques de 5 minutos)")
    con_gol = df_filt[df_filt["minuto_primer_gol"].notna()].copy()
    con_gol = con_gol[con_gol["minuto_primer_gol"] <= 45]

    if con_gol.empty:
        st.info("No hay goles registrados en 1ª parte (o están fuera de 0–45).")
    else:
        bins = list(range(0, 50, 5))
        labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]
        con_gol["bloque_5m"] = pd.cut(
            con_gol["minuto_primer_gol"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True,
        )

        distrib_df = con_gol.groupby("bloque_5m").size().reset_index(name="n_partidos")
        distrib_df["bloque_5m"] = distrib_df["bloque_5m"].astype(str)

        n_sin_gol = df_filt["minuto_primer_gol"].isna().sum()
        if n_sin_gol > 0:
            distrib_df = pd.concat(
                [distrib_df, pd.DataFrame({"bloque_5m": ["sin gol HT"], "n_partidos": [n_sin_gol]})],
                ignore_index=True,
            )

        st.bar_chart(distrib_df.set_index("bloque_5m")[["n_partidos"]], use_container_width=True)

    st.markdown("#### Percentiles del minuto del primer gol (HT)")
    serie_goles = df_filt["minuto_primer_gol"].dropna()
    serie_goles = serie_goles[serie_goles <= 45]

    if serie_goles.empty:
        st.info("No hay suficientes datos de minuto_primer_gol (0–45) para percentiles.")
    else:
        percentiles = {
            "P10": float(serie_goles.quantile(0.10)),
            "P25": float(serie_goles.quantile(0.25)),
            "P50": float(serie_goles.quantile(0.50)),
            "P75": float(serie_goles.quantile(0.75)),
            "P90": float(serie_goles.quantile(0.90)),
        }
        pct_table = pd.DataFrame({"Percentil": list(percentiles.keys()), "Minuto": [round(v, 2) for v in percentiles.values()]})
        st.table(pct_table)


# ======================================================================
# VISTA: BANCA (+ VIX)
# ======================================================================
def show_banca():
    st.markdown("### Banca – equity, ROI sobre banca y Maximum Drawdown")

    try:
        mov = fetch_banca_movimientos()
    except Exception as e:
        st.error(f"Error cargando banca_movimientos: {e}")
        return

    try:
        seg = fetch_seguimiento()
    except Exception as e:
        st.error(f"Error cargando seguimiento: {e}")
        return

    st.markdown("#### Añadir movimiento")
    with st.form("form_banca_mov"):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            fecha = st.date_input("Fecha", value=pd.Timestamp.today().date())
        with c2:
            tipo = st.selectbox("Tipo", options=["DEPOSITO", "RETIRADA", "AJUSTE"])
        with c3:
            importe = st.number_input("Importe", value=0.0, step=10.0, format="%.2f")

        comentario = st.text_input("Comentario (opcional)", value="")
        submitted = st.form_submit_button("Guardar movimiento")

        if submitted:
            try:
                insert_banca_movimiento(
                    fecha=fecha,
                    tipo=tipo,
                    importe=float(importe),
                    comentario=comentario if comentario.strip() else None,
                )
                st.success("Movimiento guardado.")
                st.rerun()
            except Exception as e:
                st.error(f"Error guardando movimiento: {e}")

    st.markdown("---")

    if mov.empty:
        st.info("No hay movimientos todavía. Inserta al menos el **depósito inicial** para poder calcular la banca.")
        return

    st.markdown("#### Configuración")
    only_real = st.checkbox("Usar solo apuestas reales (apuesta_real = SI) para el equity", value=True)

    equity_df = _compute_equity_curve(movimientos_df=mov, seguimiento_df=seg, only_real=only_real)
    if equity_df.empty:
        st.info("No hay fechas suficientes (movimientos o profits con fecha).")
        return

    banca_inicial = float(equity_df.iloc[0]["equity"]) if len(equity_df) > 0 else 0.0
    banca_actual = float(equity_df.iloc[-1]["equity"]) if len(equity_df) > 0 else 0.0

    total_profit = float(equity_df.iloc[-1]["profit_cum"]) if "profit_cum" in equity_df.columns else 0.0
    roi_banca = (total_profit / banca_inicial) if banca_inicial > 0 else None

    mdd_pct = None
    if "drawdown_pct" in equity_df.columns and equity_df["drawdown_pct"].notna().any():
        mdd_pct = float(equity_df["drawdown_pct"].min())
    mdd_abs = None
    if "drawdown_abs" in equity_df.columns and equity_df["drawdown_abs"].notna().any():
        mdd_abs = float(equity_df["drawdown_abs"].min())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Banca inicial", f"{banca_inicial:,.2f} €")
    c2.metric("Banca actual", f"{banca_actual:,.2f} €")
    c3.metric("Profit acumulado (apuestas)", f"{total_profit:,.2f} €")
    c4.metric("ROI sobre banca inicial", f"{roi_banca*100:.2f}%" if roi_banca is not None else "—")

    st.markdown("#### Maximum Drawdown")
    if mdd_pct is None:
        st.info("No se puede calcular MDD.")
    else:
        st.metric("MDD %", f"{mdd_pct*100:.2f}%")
        if mdd_abs is not None:
            st.metric("MDD €", f"{mdd_abs:,.2f} €")

    st.markdown("#### Evolución de banca (equity)")
    plot = equity_df.copy()
    plot["fecha"] = pd.to_datetime(plot["fecha"])
    st.line_chart(plot.set_index("fecha")[["equity"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### Banca + VIX (tabla cruzada por fecha)")

    try:
        vix_daily = _fetch_vix_daily_cached()
    except Exception as e:
        vix_daily = pd.DataFrame()
        st.warning(f"No se pudo leer vix_daily: {e}")

    if not vix_daily.empty and "fecha" in vix_daily.columns:
        v = vix_daily.copy()
        v["fecha"] = pd.to_datetime(v["fecha"], errors="coerce").dt.date
        v = v.dropna(subset=["fecha"])

        e = equity_df.copy()
        if "fecha" in e.columns:
            e["fecha"] = pd.to_datetime(e["fecha"], errors="coerce").dt.date

        cols_v = ["fecha"]
        for c in ["estado", "accion", "comentario", "raw_accion", "raw_comentario"]:
            if c in v.columns:
                cols_v.append(c)

        merged = e.merge(v[cols_v], on="fecha", how="left")
        st.dataframe(
            merged.sort_values("fecha", ascending=False),
            use_container_width=True
        )
    else:
        st.info("No hay VIX suficiente para cruzar con la banca.")


# ======================================================================
# VISTA: VIX (robusta: solo vix_daily + órdenes)
# ======================================================================
def show_vix():
    st.markdown("### VIX – régimen diario (SVIX / NEUTRAL / UVIX)")

    st.markdown(
        """
**Cómo leerlo:**
- **`raw_accion`** = señal del día “en bruto” (sin filtros anti-ruido).
- **`accion`** = señal final “operable” (con confirmación/cooldown si están en tu vix.py).
- **`comentario`** explica el porqué en lenguaje humano.
        """.strip()
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        start = st.date_input("Start", value=pd.Timestamp("2025-01-01").date())
    with c2:
        end = st.date_input("End (exclusivo)", value=(pd.Timestamp.today().date() + pd.Timedelta(days=1)))

    if st.button("Actualizar VIX (descargar + calcular + guardar en Supabase)"):
        try:
            run_vix_pipeline(start=str(start), end=str(end))
            st.success("VIX actualizado y guardado en Supabase (vix_daily).")
            st.rerun()
        except Exception as e:
            st.exception(e)

    st.markdown("---")

    try:
        daily = _fetch_vix_daily_cached()
    except Exception as e:
        st.error(f"Error leyendo VIX desde Supabase: {e}")
        return

    if daily.empty:
        st.info("No hay datos en vix_daily todavía.")
        return

    daily = daily.copy()
    if "fecha" in daily.columns:
        daily["fecha"] = pd.to_datetime(daily["fecha"], errors="coerce")
        daily = daily.dropna(subset=["fecha"]).sort_values("fecha")
    else:
        st.error("vix_daily no tiene columna 'fecha'.")
        return

    last = daily.iloc[-1]

    st.markdown("#### Señal del día (la importante)")
    cA, cB, cC = st.columns(3)
    cA.metric("Fecha", str(last.get("fecha").date()) if pd.notna(last.get("fecha")) else "—")
    cB.metric("Estado", str(last.get("estado", "—")))
    cC.metric("Acción", str(last.get("accion", "—")))

    st.write(f"**Comentario:** {last.get('comentario', '')}")

    if "raw_accion" in daily.columns or "raw_comentario" in daily.columns:
        st.markdown("#### Detalle (señal en bruto vs señal final)")
        st.write(f"**raw_accion:** {last.get('raw_accion', '—')}")
        st.write(f"**raw_comentario:** {last.get('raw_comentario', '')}")

    st.markdown("---")

    for col in ["vix", "vxn_vix_ratio", "vixy_ma_3", "vixy_ma_10", "vixy_ma3", "vixy_ma10"]:
        _safe_numeric(daily, col)

    st.markdown("#### Explicación rápida de los indicadores")
    st.markdown(
        """
- **VXN/VIX**: compara “miedo Nasdaq” vs “miedo S&P”. Si sube y se dispara, suele indicar stress más “tech”.
- **Contango proxy (MA corta vs MA larga del proxy de volatilidad)**: si la media corta se pone por encima de la larga,
  suele ser síntoma de tensión (carry peor / backwardation-like).
        """.strip()
    )

    if "vix" in daily.columns:
        st.markdown("#### VIX (spot)")
        st.line_chart(daily.set_index("fecha")[["vix"]], use_container_width=True)

    if "vxn_vix_ratio" in daily.columns:
        st.markdown("#### Ratio VXN/VIX")
        st.line_chart(daily.set_index("fecha")[["vxn_vix_ratio"]], use_container_width=True)

    cont_cols = []
    if "vixy_ma_3" in daily.columns and "vixy_ma_10" in daily.columns:
        cont_cols = ["vixy_ma_3", "vixy_ma_10"]
    elif "vixy_ma3" in daily.columns and "vixy_ma10" in daily.columns:
        cont_cols = ["vixy_ma3", "vixy_ma10"]

    if cont_cols:
        st.markdown("#### Contango proxy (MA corta vs MA larga)")
        st.line_chart(daily.set_index("fecha")[cont_cols], use_container_width=True)
    else:
        st.info("No hay columnas de medias móviles del proxy de volatilidad (MA3/MA10) en vix_daily.")

    st.markdown("#### Tabla (vix_daily)")
    show_cols = [c for c in [
        "fecha",
        "estado", "accion", "comentario",
        "raw_accion", "raw_comentario",
        "vix", "vxn", "vixy", "spy",
        "vxn_vix_ratio", "contango_ok", "macro_tomorrow",
        "vix_p25", "vix_p65", "vix_p85",
        "vixy_ma_3", "vixy_ma_10", "vixy_ma3", "vixy_ma10",
        "spy_ret",
    ] if c in daily.columns]

    st.dataframe(daily[show_cols].sort_values("fecha", ascending=False), use_container_width=True)

    # ==================================================================
    # ÓRDENES VIX (entrada/salida)
    # ==================================================================
    st.markdown("---")
    st.markdown("### Órdenes VIX (entrada/salida)")

    if fetch_vix_orders is None or insert_vix_order is None or update_vix_order_status is None:
        st.info("El módulo de órdenes VIX no está disponible todavía (faltan funciones en backend.vix).")
        return

    # Sugerencia: usar última señal como ayuda
    estado_hoy = str(last.get("estado", "NEUTRAL"))
    accion_hoy = str(last.get("accion", ""))
    comentario_hoy = str(last.get("comentario", ""))

    default_fecha = pd.Timestamp.today().date()
    try:
        if pd.notna(last.get("fecha")):
            default_fecha = pd.to_datetime(last.get("fecha")).date()
    except Exception:
        pass

    with st.expander("➕ Crear orden", expanded=True):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            fecha_ord = st.date_input("Fecha orden", value=default_fecha)
        with c2:
            ticker = st.selectbox("Ticker", options=["SVIX", "UVIX"], index=0)
        with c3:
            side = st.selectbox("Side", options=["BUY", "SELL"], index=0)
        with c4:
            status = st.selectbox("Status", options=["PLANNED", "EXECUTED", "CANCELLED"], index=0)

        c5, c6, c7 = st.columns([1, 1, 2])
        with c5:
            qty = st.number_input("Cantidad (qty)", min_value=0.0, value=1.0, step=1.0)
        with c6:
            price = st.number_input("Precio (opcional)", min_value=0.0, value=0.0, step=0.01)
        with c7:
            notes_default = f"Auto: estado={estado_hoy} | accion={accion_hoy}"
            notes = st.text_input("Notas (opcional)", value=notes_default)

        st.caption(f"Comentario señal (hoy): {comentario_hoy}")

        if st.button("Guardar orden"):
            try:
                insert_vix_order(
                    fecha=fecha_ord,
                    ticker=ticker,
                    side=side,
                    qty=float(qty),
                    price=(float(price) if float(price) > 0 else None),
                    status=status,
                    notes=notes if notes.strip() else None,
                    estado_signal=estado_hoy,
                )
                st.success("Orden guardada en Supabase (vix_orders).")
                st.rerun()
            except Exception as e:
                st.error(f"Error guardando orden: {e}")

    # Listado + acciones rápidas
    try:
        orders = fetch_vix_orders(limit=300)
    except Exception as e:
        st.error(f"Error leyendo vix_orders: {e}")
        orders = pd.DataFrame()

    if orders.empty:
        st.info("No hay órdenes todavía.")
    else:
        st.markdown("#### Últimas órdenes")
        show_cols_o = [c for c in ["id","fecha","estado_signal","ticker","side","qty","price","status","notes","created_at","updated_at"] if c in orders.columns]
        st.dataframe(orders[show_cols_o], use_container_width=True)

        st.markdown("#### Actualizar estado de una orden")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

        default_id = 1
        try:
            if "id" in orders.columns and len(orders) > 0:
                default_id = int(orders.iloc[0]["id"])
        except Exception:
            default_id = 1

        with c1:
            order_id = st.number_input("Order ID", min_value=1, value=default_id, step=1)
        with c2:
            new_status = st.selectbox("Nuevo status", options=["PLANNED","EXECUTED","CANCELLED"], index=1)
        with c3:
            new_price = st.number_input("Precio ejecución (opcional)", min_value=0.0, value=0.0, step=0.01)
        with c4:
            new_notes = st.text_input("Notas actualización (opcional)", value="")

        if st.button("Actualizar orden"):
            try:
                update_vix_order_status(
                    order_id=int(order_id),
                    status=new_status,
                    price=(float(new_price) if float(new_price) > 0 else None),
                    notes=(new_notes if new_notes.strip() else None),
                )
                st.success("Orden actualizada.")
                st.rerun()
            except Exception as e:
                st.error(f"Error actualizando orden: {e}")


# ======================================================================
# INTERFAZ PRINCIPAL
# ======================================================================
def main():
    st.set_page_config(page_title="Selector de Partidos HT/FT", layout="wide")

    st.sidebar.title("Navegación")
    modo = st.sidebar.radio(
        "Selecciona modo",
        options=[
            "Selector de partidos",
            "Gestión de apuestas",
            "Estadísticas ROI",
            "Supervivencia & Convexidad",
            "Banca",
            "VIX",
        ],
    )

    if modo == "Selector de partidos":
        st.title("Selector de partidos HT/FT – Modelo")
        show_selector()
    elif modo == "Gestión de apuestas":
        st.title("Gestión de apuestas – Seguimiento")
        show_gestion()
    elif modo == "Estadísticas ROI":
        st.title("Estadísticas de ROI")
        show_stats()
    elif modo == "Supervivencia & Convexidad":
        st.title("Supervivencia & Convexidad")
        show_supervivencia_convexidad()
    elif modo == "Banca":
        st.title("Banca")
        show_banca()
    else:
        st.title("VIX")
        show_vix()


if __name__ == "__main__":
    main()
