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

# ---------- CARGA HISTÓRICO (CACHEADO) ----------
@st.cache_data(show_spinner="Cargando histórico y calculando estadísticas…")
def _load_hist_and_stats():
    hist = load_historical_data("data")
    team_stats, div_stats = compute_team_and_league_stats(hist)
    return hist, team_stats, div_stats


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

    # ✅ CAMBIO: mostrar también "Buena" (además de Ideal y Buena filtrada)
    # - PickType suele estar solo en Ideal/Buena filtrada
    # - Buena suele venir en MatchClass
    picks = scored[
        scored["MatchClass"].isin(["Ideal", "Buena", "Buena filtrada"])
        | scored["PickType"].notna()
    ].copy()

    st.markdown("### 2. Resultados del modelo")

    if picks.empty:
        st.warning("Ningún partido cumple los filtros de visualización para este fixture.")
        return

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
        "apuesta_real",
        "minuto_primer_gol",
        "pct_minuto_primer_gol",
        "estrategia",
        "raroc",
    ]

    if "fecha" in filtered.columns:
        filtered = filtered.sort_values(["fecha", "hora", "division", "home_team"])

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

        # --- estrategia ---
        estrategia_actual = (row_sel.get("estrategia") or "Convexidad")
        estrategia = st.selectbox(
            "Estrategia",
            options=["Convexidad", "Spread Attack"],
            index=0 if estrategia_actual == "Convexidad" else 1,
        )

        stake_btts_no = st.number_input("Stake BTTS NO", value=float(row_sel["stake_btts_no"]) if pd.notna(row_sel.get("stake_btts_no")) else 0.0, step=1.0)
        stake_u35 = st.number_input("Stake Under 3.5", value=float(row_sel["stake_u35"]) if pd.notna(row_sel.get("stake_u35")) else 0.0, step=1.0)
        stake_1_1 = st.number_input("Stake marcador 1-1", value=float(row_sel["stake_1_1"]) if pd.notna(row_sel.get("stake_1_1")) else 0.0, step=1.0)

        close_minute_global = st.number_input("Minuto de cierre global", value=int(row_sel["close_minute_global"]) if pd.notna(row_sel.get("close_minute_global")) else 0, step=1)
        close_minute_1_1 = st.number_input("Minuto de cierre 1-1", value=int(row_sel["close_minute_1_1"]) if pd.notna(row_sel.get("close_minute_1_1")) else 0, step=1)

        odds_btts_no_init = st.number_input("Cuota inicial BTTS NO", value=float(row_sel["odds_btts_no_init"]) if pd.notna(row_sel.get("odds_btts_no_init")) else 0.0, step=0.01)
        odds_u35_init = st.number_input("Cuota inicial Under 3.5", value=float(row_sel["odds_u35_init"]) if pd.notna(row_sel.get("odds_u35_init")) else 0.0, step=0.01)
        odds_1_1_init = st.number_input("Cuota inicial 1-1", value=float(row_sel["odds_1_1_init"]) if pd.notna(row_sel.get("odds_1_1_init")) else 0.0, step=0.01)

        profit_euros = st.number_input("Profit (€)", value=float(row_sel["profit_euros"]) if pd.notna(row_sel.get("profit_euros")) else 0.0, step=1.0)

        # minuto primer gol
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

        # ROI calculado automáticamente
        total_stake = stake_btts_no + stake_u35 + stake_1_1
        if total_stake > 0:
            roi_calc = profit_euros / total_stake
            st.write(f"ROI calculado: **{roi_calc*100:.2f}%** (profit / suma de stakes)")
        else:
            roi_calc = None
            st.write("ROI calculado: — (faltan stakes o profit)")

        # RAROC (% por minuto expuesto) usando close_minute_global, sin /45
        raroc_calc = None
        if roi_calc is not None and close_minute_global and close_minute_global > 0:
            raroc_calc = (roi_calc * 100.0) / float(close_minute_global)
            st.write(f"RAROC (ROI% / minuto): **{raroc_calc:.4f}% por minuto**")
        else:
            st.write("RAROC: — (necesita ROI y close_minute_global > 0). Para cierres en 0 min, no es comparable (∞/N/A).")

        apuesta_real_actual = row_sel.get("apuesta_real") or "NO"
        apuesta_real = st.selectbox("¿Apuesta real?", options=["SI", "NO"], index=0 if apuesta_real_actual == "SI" else 1)

        submitted = st.form_submit_button("Guardar cambios (formulario)")

        if submitted:
            cambios = {
                "estrategia": estrategia,
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
                "raroc": raroc_calc,
            }

            try:
                update_seguimiento_row(selected_id, cambios)
                st.success(f"Registro ID {selected_id} actualizado correctamente.")
            except Exception as e:
                st.error(f"Error actualizando (formulario) en Supabase: {e}")


# ======================================================================
# VISTA: ESTADÍSTICAS ROI
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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_stake"] = (
        df.get("stake_btts_no", 0).fillna(0)
        + df.get("stake_u35", 0).fillna(0)
        + df.get("stake_1_1", 0).fillna(0)
    )

    df["roi_calc"] = pd.NA
    mask_valid = (df["total_stake"] > 0) & df["profit_euros"].notna()
    df.loc[mask_valid, "roi_calc"] = df.loc[mask_valid, "profit_euros"] / df.loc[mask_valid, "total_stake"]

    apuesta_real_opts = sorted([x for x in df.get("apuesta_real", pd.Series()).dropna().unique()])
    if apuesta_real_opts:
        ar_sel = st.multiselect("Filtrar por tipo de apuesta", options=apuesta_real_opts, default=apuesta_real_opts)
        df = df[df["apuesta_real"].isin(ar_sel)]

    if df.empty:
        st.warning("No hay registros tras aplicar filtros.")
        return

    total_profit = df["profit_euros"].fillna(0).sum()
    total_stake_sum = df["total_stake"].fillna(0).sum()
    roi_global = (total_profit / total_stake_sum) if total_stake_sum > 0 else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total profit (€)", f"{total_profit:,.2f}")
    with col2:
        st.metric("Total stake (€)", f"{total_stake_sum:,.2f}")
    with col3:
        st.metric("ROI global", f"{roi_global*100:.2f}%" if roi_global is not None else "—")

    st.markdown("### Gráficos de ROI (en %)")

    if "fecha" in df.columns and df["fecha"].notna().any():
        df_time = df[df["fecha"].notna()].sort_values("fecha").copy()
        df_time["profit_acum"] = df_time["profit_euros"].fillna(0).cumsum()
        df_time["stake_acum"] = df_time["total_stake"].fillna(0).cumsum()
        df_time["roi_acum"] = pd.NA
        mask_ok = df_time["stake_acum"] > 0
        df_time.loc[mask_ok, "roi_acum"] = df_time.loc[mask_ok, "profit_acum"] / df_time.loc[mask_ok, "stake_acum"]
        df_time["roi_acum_pct"] = pd.to_numeric(df_time["roi_acum"], errors="coerce") * 100.0

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
# VISTA: SUPERVIVENCIA & CONVEXIDAD
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
        st.warning("La tabla 'seguimiento' no tiene la columna 'minuto_primer_gol'. Añádela en Supabase.")
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
        return

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
    else:
        st.title("Supervivencia & Convexidad")
        show_supervivencia_convexidad()


if __name__ == "__main__":
    main()
