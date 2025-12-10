# app/main.py

import os
import sys

import numpy as np
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


# ======================================================================
# CARGA HISTÓRICO (CACHEADO)
# ======================================================================

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

    # Normalizamos fecha para filtrar
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # ----------------- FILTROS -----------------
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

        # Filtro por equipo (aparece como local o visitante)
        equipos = sorted(
            set(df.get("home_team", pd.Series()).dropna().unique())
            | set(df.get("away_team", pd.Series()).dropna().unique())
        )
        if equipos:
            equipos_filter = st.multiselect(
                "Filtrar por equipo (local o visitante)",
                options=equipos,
                default=[],
            )
        else:
            equipos_filter = []

        # Filtro por apuesta_real
        if "apuesta_real" in df.columns:
            ar_values = sorted([x for x in df["apuesta_real"].dropna().unique()])
            if ar_values:
                apuesta_real_filter = st.multiselect(
                    "Filtrar por apuesta_real (SI/NO)",
                    options=ar_values,
                    default=ar_values,
                )
            else:
                apuesta_real_filter = []
        else:
            apuesta_real_filter = []

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

    if equipos_filter:
        mask &= df["home_team"].isin(equipos_filter) | df["away_team"].isin(equipos_filter)

    if apuesta_real_filter and "apuesta_real" in df.columns:
        mask &= df["apuesta_real"].isin(apuesta_real_filter)

    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No hay registros que cumplan los filtros.")
        return

    st.write(f"Registros filtrados: **{len(filtered)}**")

    # Columnas editables (incluimos apuesta_real)
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
    ]

    # Ordenamos por fecha/hora para que sea más legible
    if "fecha" in filtered.columns:
        filtered = filtered.sort_values(["fecha", "hora", "division", "home_team"])

    # ----------------- EDICIÓN RÁPIDA (TABLA) -----------------
    st.markdown("#### Edición rápida (tabla)")

    edited = st.data_editor(
        filtered,
        use_container_width=True,
        key="editor_seguimiento",
        hide_index=True,
    )

    if st.button("Guardar cambios en Supabase (tabla)"):
        try:
            updated = update_seguimiento_from_df(
                original_df=filtered,
                edited_df=edited,
                editable_cols=editable_cols,
            )
            st.success(f"Se han actualizado {updated} filas en la tabla 'seguimiento'.")
        except Exception as e:
            st.error(f"Error actualizando en Supabase: {e}")

    # ----------------- EDICIÓN DETALLADA (FORMULARIO) -----------------
    st.markdown("---")
    st.markdown("#### Edición detallada (modo formulario)")

    if "id" not in filtered.columns:
        st.warning("No hay columna 'id' en los datos, no se puede usar el modo formulario.")
        return

    # Construimos opciones legibles para seleccionar registro
    opciones = []
    for _, row in filtered.iterrows():
        etiqueta = (
            f"ID {row['id']} - {row.get('fecha', '')} - {row.get('division', '')} - "
            f"{row.get('home_team', '')} vs {row.get('away_team', '')}"
        )
        opciones.append((int(row["id"]), etiqueta))

    if not opciones:
        st.info("No hay registros para mostrar en formulario.")
        return

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

        stake_btts_no = st.number_input(
            "Stake BTTS NO",
            value=float(row_sel["stake_btts_no"]) if pd.notna(row_sel.get("stake_btts_no")) else 0.0,
            step=1.0,
        )
        stake_u35 = st.number_input(
            "Stake Under 3.5",
            value=float(row_sel["stake_u35"]) if pd.notna(row_sel.get("stake_u35")) else 0.0,
            step=1.0,
        )
        stake_1_1 = st.number_input(
            "Stake marcador 1-1",
            value=float(row_sel["stake_1_1"]) if pd.notna(row_sel.get("stake_1_1")) else 0.0,
            step=1.0,
        )

        close_minute_global = st.number_input(
            "Minuto de cierre global",
            value=int(row_sel["close_minute_global"]) if pd.notna(row_sel.get("close_minute_global")) else 0,
            step=1,
        )
        close_minute_1_1 = st.number_input(
            "Minuto de cierre 1-1",
            value=int(row_sel["close_minute_1_1"]) if pd.notna(row_sel.get("close_minute_1_1")) else 0,
            step=1,
        )

        odds_btts_no_init = st.number_input(
            "Cuota inicial BTTS NO",
            value=float(row_sel["odds_btts_no_init"]) if pd.notna(row_sel.get("odds_btts_no_init")) else 0.0,
            step=0.01,
        )
        odds_u35_init = st.number_input(
            "Cuota inicial Under 3.5",
            value=float(row_sel["odds_u35_init"]) if pd.notna(row_sel.get("odds_u35_init")) else 0.0,
            step=0.01,
        )
        odds_1_1_init = st.number_input(
            "Cuota inicial 1-1",
            value=float(row_sel["odds_1_1_init"]) if pd.notna(row_sel.get("odds_1_1_init")) else 0.0,
            step=0.01,
        )

        profit_euros = st.number_input(
            "Profit (€)",
            value=float(row_sel["profit_euros"]) if pd.notna(row_sel.get("profit_euros")) else 0.0,
            step=1.0,
        )

        # Minuto primer gol
        minuto_primer_gol = st.number_input(
            "Minuto del primer gol (HT)",
            value=int(row_sel["minuto_primer_gol"]) if pd.notna(row_sel.get("minuto_primer_gol")) else 0,
            step=1,
            min_value=0,
            max_value=45,
        )

        # ROI calculado automáticamente: profit / suma de stakes
        total_stake = stake_btts_no + stake_u35 + stake_1_1
        if total_stake > 0:
            roi_calc = profit_euros / total_stake
            st.write(f"ROI calculado: **{roi_calc:.3f}** (profit / suma de stakes)")
        else:
            roi_calc = None
            st.write("ROI calculado: — (faltan stakes o profit)")

        apuesta_real_actual = row_sel.get("apuesta_real") or "NO"
        apuesta_real = st.selectbox(
            "¿Apuesta real?",
            options=["SI", "NO"],
            index=0 if apuesta_real_actual == "SI" else 1,
        )

        # Percentil mostrado (si existe)
        pct_actual = row_sel.get("pct_minuto_primer_gol")
        if pd.notna(pct_actual):
            st.write(f"Percentil minuto primer gol guardado: **{pct_actual:.1f}**")
        else:
            st.write("Percentil minuto primer gol guardado: —")

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
                "minuto_primer_gol": minuto_primer_gol if minuto_primer_gol > 0 else None,
            }

            # Solo enviamos ROI si se ha podido calcular
            if roi_calc is not None:
                cambios["roi"] = roi_calc
            else:
                cambios["roi"] = None

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

    # Normalizamos fecha
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Calculamos total_stake y roi_calculado (independiente del campo roi guardado)
    for col in ["stake_btts_no", "stake_u35", "stake_1_1", "profit_euros"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_stake"] = (
        df.get("stake_btts_no", 0).fillna(0)
        + df.get("stake_u35", 0).fillna(0)
        + df.get("stake_1_1", 0).fillna(0)
    )

    df["roi_calc"] = None
    mask_valid = (df["total_stake"] > 0) & df["profit_euros"].notna()
    df.loc[mask_valid, "roi_calc"] = df.loc[mask_valid, "profit_euros"] / df.loc[mask_valid, "total_stake"]

    # Filtro por apuesta_real (para separar reales de simuladas)
    apuesta_real_opts = sorted([x for x in df.get("apuesta_real", pd.Series()).dropna().unique()])
    if apuesta_real_opts:
        ar_sel = st.multiselect(
            "Filtrar por tipo de apuesta",
            options=apuesta_real_opts,
            default=apuesta_real_opts,
        )
        df = df[df["apuesta_real"].isin(ar_sel)]

    if df.empty:
        st.warning("No hay registros tras aplicar el filtro de apuesta_real.")
        return

    # ----------------- TABLAS DE ROI -----------------

    # ROI global
    total_profit = df["profit_euros"].fillna(0).sum()
    total_stake_sum = df["total_stake"].fillna(0).sum()
    roi_global = total_profit / total_stake_sum if total_stake_sum > 0 else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total profit (€)", f"{total_profit:,.2f}")
    with col2:
        st.metric("Total stake (€)", f"{total_stake_sum:,.2f}")
    with col3:
        if roi_global is not None:
            st.metric("ROI global", f"{roi_global*100:.2f} %")
        else:
            st.metric("ROI global", "—")

    st.markdown("#### ROI por tipo de apuesta (apuesta_real)")

    if "apuesta_real" in df.columns:
        grp_ar = (
            df.groupby("apuesta_real")
            .apply(lambda g: pd.Series({
                "profit_total": g["profit_euros"].fillna(0).sum(),
                "stake_total": g["total_stake"].fillna(0).sum(),
            }))
            .reset_index()
        )
        grp_ar["roi"] = grp_ar.apply(
            lambda r: r["profit_total"] / r["stake_total"] if r["stake_total"] > 0 else None,
            axis=1,
        )
        grp_ar["roi_pct"] = grp_ar["roi"] * 100
        st.dataframe(grp_ar, use_container_width=True)
    else:
        st.write("No existe columna 'apuesta_real' en los datos.")

    st.markdown("#### ROI por división")

    if "division" in df.columns:
        grp_div = (
            df.groupby("division")
            .apply(lambda g: pd.Series({
                "profit_total": g["profit_euros"].fillna(0).sum(),
                "stake_total": g["total_stake"].fillna(0).sum(),
                "n_apuestas": len(g),
            }))
            .reset_index()
        )
        grp_div["roi"] = grp_div.apply(
            lambda r: r["profit_total"] / r["stake_total"] if r["stake_total"] > 0 else None,
            axis=1,
        )
        grp_div["roi_pct"] = grp_div["roi"] * 100
        st.dataframe(grp_div.sort_values("roi", ascending=False), use_container_width=True)
    else:
        st.write("No existe columna 'division' en los datos.")

    st.markdown("#### ROI por PickType")

    if "pick_type" in df.columns:
        grp_pick = (
            df.groupby("pick_type")
            .apply(lambda g: pd.Series({
                "profit_total": g["profit_euros"].fillna(0).sum(),
                "stake_total": g["total_stake"].fillna(0).sum(),
                "n_apuestas": len(g),
            }))
            .reset_index()
        )
        grp_pick["roi"] = grp_pick.apply(
            lambda r: r["profit_total"] / r["stake_total"] if r["stake_total"] > 0 else None,
            axis=1,
        )
        grp_pick["roi_pct"] = grp_pick["roi"] * 100
        st.dataframe(grp_pick.sort_values("roi", ascending=False), use_container_width=True)
    else:
        st.write("No existe columna 'pick_type' en los datos.")

    # ----------------- GRÁFICOS DE ROI -----------------

    st.markdown("### Gráficos de ROI")

    # ROI acumulado en el tiempo
    if "fecha" in df.columns and df["fecha"].notna().any():
        df_time = df[df["fecha"].notna()].sort_values("fecha").copy()
        df_time["profit_acum"] = df_time["profit_euros"].fillna(0).cumsum()
        df_time["stake_acum"] = df_time["total_stake"].fillna(0).cumsum()
        df_time["roi_acum"] = df_time.apply(
            lambda r: r["profit_acum"] / r["stake_acum"] if r["stake_acum"] > 0 else None,
            axis=1,
        )

        st.markdown("#### ROI acumulado en el tiempo (sobre importe apostado)")
        st.line_chart(
            df_time.set_index("fecha")[["roi_acum"]],
            use_container_width=True,
        )
    else:
        st.write("No hay fechas válidas para dibujar ROI acumulado.")

    # ROI por división (barras)
    if "division" in df.columns:
        st.markdown("#### ROI por división (barras)")
        grp_div_plot = (
            df.groupby("division")
            .apply(lambda g: pd.Series({
                "profit_total": g["profit_euros"].fillna(0).sum(),
                "stake_total": g["total_stake"].fillna(0).sum(),
            }))
            .reset_index()
        )
        grp_div_plot["roi"] = grp_div_plot.apply(
            lambda r: r["profit_total"] / r["stake_total"] if r["stake_total"] > 0 else 0,
            axis=1,
        )
        grp_div_plot["roi_pct"] = grp_div_plot["roi"] * 100
        st.bar_chart(
            grp_div_plot.set_index("division")[["roi_pct"]],
            use_container_width=True,
        )


# ======================================================================
# VISTA: SUPERVIVENCIA & CONVEXIDAD
# ======================================================================

def show_supervivencia_convexidad():
    st.markdown("### Supervivencia & Convexidad")

    try:
        df = fetch_seguimiento()
    except Exception as e:
        st.error(f"Error cargando datos de seguimiento: {e}")
        return

    if df.empty:
        st.info("Todavía no hay registros en 'seguimiento'.")
        return

    # Normalizamos fecha
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Aseguramos tipos numéricos
    if "minuto_primer_gol" in df.columns:
        df["minuto_primer_gol"] = pd.to_numeric(df["minuto_primer_gol"], errors="coerce")

    # ----------------- FILTROS BÁSICOS -----------------
    with st.expander("Filtros para análisis de supervivencia"):
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

        # División
        divisiones = sorted([x for x in df.get("division", pd.Series()).dropna().unique()])
        if divisiones:
            div_filter = st.multiselect(
                "Filtrar por división",
                options=divisiones,
                default=divisiones,
            )
        else:
            div_filter = []

        # PickType
        pick_types = sorted([x for x in df.get("pick_type", pd.Series()).dropna().unique()])
        if pick_types:
            pick_filter = st.multiselect(
                "Filtrar por PickType",
                options=pick_types,
                default=pick_types,
            )
        else:
            pick_filter = []

        # Apuesta real
        if "apuesta_real" in df.columns:
            ar_values = sorted([x for x in df["apuesta_real"].dropna().unique()])
            if ar_values:
                apuesta_real_filter = st.multiselect(
                    "Filtrar por apuesta_real (SI/NO)",
                    options=ar_values,
                    default=ar_values,
                )
            else:
                apuesta_real_filter = []
        else:
            apuesta_real_filter = []

    # Aplicamos filtros
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

    df_filt = df[mask].copy()

    if df_filt.empty:
        st.warning("No hay registros que cumplan los filtros para supervivencia.")
        return

    # ================= SUPERVIVENCIA =================
    st.markdown("#### Curva de supervivencia (probabilidad de llegar sin gol hasta t)")

    if "minuto_primer_gol" not in df_filt.columns:
        st.warning("No existe la columna 'minuto_primer_gol' en la tabla 'seguimiento'.")
    else:
        m = df_filt["minuto_primer_gol"]

        minutos = list(range(0, 46))
        surv = []
        for t in minutos:
            # Consideramos sin gol (NaN) como > 45 (censura a 45')
            surv.append(((m.isna()) | (m > t)).mean())

        surv_df = pd.DataFrame({"minuto": minutos, "supervivencia": surv})
        st.line_chart(surv_df.set_index("minuto"), use_container_width=True)

    st.markdown("#### Distribución del minuto del primer gol (bloques de 5 minutos)")

    if "minuto_primer_gol" in df_filt.columns:
        df_goals = df_filt[df_filt["minuto_primer_gol"].notna()].copy()

        if df_goals.empty:
            st.info("No hay registros con 'minuto_primer_gol' informado.")
        else:
            # Bins de 5 minutos: 0-5, 5-10, ..., 40-45
            bins = np.arange(0, 50, 5)  # 0,5,10,...45
            df_goals["bloque_5m"] = pd.cut(
                df_goals["minuto_primer_gol"],
                bins=bins,
                right=True,
                include_lowest=True,
            )

            distrib_df = (
                df_goals.groupby("bloque_5m")
                .size()
                .reset_index(name="n_partidos")
            )

            # --- FIX: convertir Interval a string para Streamlit ---
            distrib_df["bloque_5m_label"] = distrib_df["bloque_5m"].astype(str)

            st.dataframe(distrib_df, use_container_width=True)

            st.bar_chart(
                distrib_df.set_index("bloque_5m_label")[["n_partidos"]],
                use_container_width=True,
            )

    # ================= PERCENTIL MINUTO PRIMER GOL =================

    st.markdown("#### Percentil del minuto del primer gol")

    if "minuto_primer_gol" in df_filt.columns:
        df_pct = df_filt[df_filt["minuto_primer_gol"].notna()].copy()

        if df_pct.empty:
            st.info("No hay registros con minuto_primer_gol para calcular percentiles.")
        else:
            # Aseguramos existencia de columna pct_minuto_primer_gol
            if "pct_minuto_primer_gol" not in df_pct.columns:
                df_pct["pct_minuto_primer_gol"] = np.nan

            # Percentil empírico dentro del subconjunto filtrado
            df_pct["pct_calc"] = df_pct["minuto_primer_gol"].rank(pct=True) * 100

            st.dataframe(
                df_pct[["id", "fecha", "division", "home_team", "away_team",
                        "minuto_primer_gol", "pct_minuto_primer_gol", "pct_calc"]],
                use_container_width=True,
            )

            if st.button("Guardar percentiles calculados en Supabase"):
                n_updates = 0
                for _, row in df_pct.iterrows():
                    row_id = int(row["id"])
                    pct_guardado = row.get("pct_minuto_primer_gol")
                    pct_nuevo = row.get("pct_calc")

                    if pd.isna(pct_nuevo):
                        continue

                    # Solo actualizamos si no hay valor guardado o es distinto
                    if pd.isna(pct_guardado) or abs(pct_guardado - pct_nuevo) > 1e-6:
                        try:
                            update_seguimiento_row(row_id, {"pct_minuto_primer_gol": float(pct_nuevo)})
                            n_updates += 1
                        except Exception as e:
                            st.error(f"Error actualizando percentil para id={row_id}: {e}")
                            break

                st.success(f"Percentiles actualizados para {n_updates} registros.")


# ======================================================================
# INTERFAZ PRINCIPAL
# ======================================================================

def main():
    st.set_page_config(
        page_title="Selector de Partidos HT/FT",
        layout="wide",
    )

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
