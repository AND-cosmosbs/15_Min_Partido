# backend/seguimiento.py

from typing import List, Dict
import pandas as pd

from .supabase_client import supabase


def _to_int_or_none(value):
    """Convierte a int o devuelve None si no se puede."""
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def insert_seguimiento_from_picks(picks_df: pd.DataFrame) -> None:
    """
    Inserta en la tabla 'seguimiento' las filas de picks_df.

    picks_df debe contener al menos:
      - Date, Time, Div, HomeTeam, AwayTeam
      - B365H, B365D, B365A
      - L_score, H_T_score, A_T_score, MatchScore, MatchClass, PickType
    """

    if picks_df.empty:
        return

    records: List[Dict] = []

    for _, r in picks_df.iterrows():

        # ----- Fecha como string ISO "YYYY-MM-DD" -----
        fecha_str = None
        if "Date" in r and pd.notna(r["Date"]):
            try:
                fecha_str = pd.to_datetime(r["Date"]).date().isoformat()
            except Exception:
                fecha_str = None

        # ----- Hora como texto -----
        hora = None
        if "Time" in r and pd.notna(r["Time"]):
            hora = str(r["Time"])

        # Campos INT: los forzamos a int o None
        l_score = _to_int_or_none(r.get("L_score"))
        h_t_score = _to_int_or_none(r.get("H_T_score"))
        a_t_score = _to_int_or_none(r.get("A_T_score"))
        match_score = _to_int_or_none(r.get("MatchScore"))

        match_class = r.get("MatchClass")
        pick_type = r.get("PickType")

        # ----- Registro final -----
        rec: Dict = {
            "fecha": fecha_str,
            "hora": hora,

            # En tu BD hay 'division' y también 'div' (NOT NULL).
            # Rellenamos ambas con el mismo valor del campo Div de Football-Data.
            "division": r.get("Div"),
            "div": r.get("Div"),

            "home_team": r.get("HomeTeam"),
            "away_team": r.get("AwayTeam"),

            # Cuotas iniciales (1X2)
            "b365h": r.get("B365H"),
            "b365d": r.get("B365D"),
            "b365a": r.get("B365A"),

            # Scores del modelo (INT en la BD)
            "l_score": l_score,
            "h_t_score": h_t_score,
            "a_t_score": a_t_score,
            "match_score": match_score,

            # Clasificación del modelo
            "match_class": match_class,
            "pick_type": pick_type,

            # Campo 'model_class' (NOT NULL en tu BD)
            "model_class": match_class,

            # Por defecto, cuando se registra desde el modelo lo marcamos como NO real
            "apuesta_real": "NO",
        }

        records.append(rec)

    # ----- Inserción en Supabase -----
    resp = supabase.table("seguimiento").insert(records).execute()

    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en seguimiento: {resp.error}")


def fetch_seguimiento() -> pd.DataFrame:
    """
    Devuelve toda la tabla 'seguimiento' como DataFrame.
    """
    resp = supabase.table("seguimiento").select("*").execute()

    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo seguimiento: {resp.error}")

    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)

    return df


def update_seguimiento_from_df(
    original_df: pd.DataFrame,
    edited_df: pd.DataFrame,
    editable_cols: List[str],
) -> int:
    """
    Actualiza la tabla 'seguimiento' comparando original_df y edited_df.

    - original_df y edited_df deben tener una columna 'id'.
    - Solo se actualizan las columnas en editable_cols.
    - Devuelve el número de filas que se han actualizado.
    """

    if "id" not in original_df.columns or "id" not in edited_df.columns:
        raise ValueError("Ambos DataFrames deben tener columna 'id'.")

    # Nos aseguramos de que id sea índice
    orig = original_df.set_index("id")
    edit = edited_df.set_index("id")

    updated_rows = 0

    for row_id in edit.index:
        if row_id not in orig.index:
            continue

        row_new = edit.loc[row_id]
        row_old = orig.loc[row_id]

        changes: Dict = {}

        for col in editable_cols:
            if col not in edit.columns:
                continue

            val_new = row_new.get(col)
            val_old = row_old.get(col)

            # Normalizamos NaN/None
            is_new_nan = pd.isna(val_new)
            is_old_nan = pd.isna(val_old)

            if is_new_nan and is_old_nan:
                continue  # igual (ambos vacío)

            if (is_new_nan and not is_old_nan) or (not is_new_nan and is_old_nan):
                # Uno vacío y otro no → cambio
                changes[col] = None if is_new_nan else val_new
            else:
                # Ambos no NaN: comparamos valores
                if val_new != val_old:
                    changes[col] = val_new

        if not changes:
            continue  # nada que actualizar en esta fila

        # Forzamos ints en los minutos de cierre para evitar problemas de tipos
        for minute_col in ["close_minute_global", "close_minute_1_1"]:
            if minute_col in changes:
                changes[minute_col] = _to_int_or_none(changes[minute_col])

        # También normalizamos minuto_primer_gol si viene en cambios
        if "minuto_primer_gol" in changes:
            changes["minuto_primer_gol"] = _to_int_or_none(changes["minuto_primer_gol"])

        # Ejecutamos update en Supabase
        resp = supabase.table("seguimiento").update(changes).eq("id", int(row_id)).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Error actualizando id={row_id} en seguimiento: {resp.error}")

        updated_rows += 1

    return updated_rows


def update_seguimiento_row(row_id: int, changes: Dict) -> None:
    """
    Actualiza una sola fila de 'seguimiento' por id, con los campos dados en changes.
    """
    # Normalizamos los campos que deben ser INT
    for minute_col in ["close_minute_global", "close_minute_1_1", "minuto_primer_gol"]:
        if minute_col in changes:
            changes[minute_col] = _to_int_or_none(changes[minute_col])

    resp = supabase.table("seguimiento").update(changes).eq("id", int(row_id)).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error actualizando id={row_id} en seguimiento: {resp.error}")


def compute_and_update_pct_minuto_primer_gol(df: pd.DataFrame) -> int:
    """
    Calcula el percentil del minuto_primer_gol entre todos los partidos
    con pick_type Ideal / Buena filtrada y minuto_primer_gol informado,
    y lo guarda en la columna pct_minuto_primer_gol de la tabla 'seguimiento'.

    Devuelve el número de filas actualizadas.
    """
    if df.empty:
        return 0

    if "id" not in df.columns or "minuto_primer_gol" not in df.columns:
        return 0

    # Trabajamos con una copia
    work = df.copy()

    # Normalizamos minuto_primer_gol a numérico
    work["minuto_primer_gol"] = pd.to_numeric(
        work["minuto_primer_gol"],
        errors="coerce",
    )

    # Filtramos Ideal + Buena filtrada con minuto_primer_gol válido
    mask = work["minuto_primer_gol"].notna()
    if "pick_type" in work.columns:
        mask &= work["pick_type"].isin(["Ideal", "Buena filtrada"])

    subset = work[mask].copy()
    if subset.empty:
        return 0

    # Percentil (0-1) del minuto_primer_gol dentro del subconjunto
    # rank(pct=True) da valores entre 0 y 1
    subset["pct"] = subset["minuto_primer_gol"].rank(method="average", pct=True)

    updated = 0

    for _, row in subset.iterrows():
        row_id = row["id"]
        pct_val = float(row["pct"])

        resp = supabase.table("seguimiento").update(
            {"pct_minuto_primer_gol": pct_val}
        ).eq("id", int(row_id)).execute()

        if getattr(resp, "error", None):
            raise RuntimeError(
                f"Error actualizando pct_minuto_primer_gol para id={row_id}: {resp.error}"
            )
        updated += 1

    return updated
