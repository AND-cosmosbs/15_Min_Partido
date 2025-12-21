# backend/seguimiento.py

from typing import List, Dict
import pandas as pd

from .supabase_client import supabase


def _to_int_or_none(value):
    """Convierte a int o devuelve None si no se puede."""
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _to_float_or_none(value):
    """Convierte a float o devuelve None si no se puede."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def insert_seguimiento_from_picks(picks_df: pd.DataFrame) -> None:
    """
    Inserta en la tabla 'seguimiento' las filas de picks_df.

    picks_df debe contener al menos:
      - Date, Time, Div, HomeTeam, AwayTeam
      - B365H, B365D, B365A
      - L_score, H_T_score, A_T_score, MatchScore, MatchClass, PickType (PickType puede venir vacío)
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

        # Scores INT
        l_score = _to_int_or_none(r.get("L_score"))
        h_t_score = _to_int_or_none(r.get("H_T_score"))
        a_t_score = _to_int_or_none(r.get("A_T_score"))
        match_score = _to_int_or_none(r.get("MatchScore"))

        match_class = r.get("MatchClass")  # Ideal / Buena / Borderline / Descartar
        pick_type_raw = r.get("PickType")  # Ideal / Buena filtrada (a veces vacío)
        pick_type = pick_type_raw if (pick_type_raw is not None and str(pick_type_raw).strip() != "" and pd.notna(pick_type_raw)) else match_class

        # Fallback final por si también viniera vacío
        if pick_type is None or (isinstance(pick_type, str) and pick_type.strip() == "") or pd.isna(pick_type):
            pick_type = "Buena"

        # Cuotas iniciales numéricas (si vienen como texto, también las acepta)
        b365h = _to_float_or_none(r.get("B365H"))
        b365d = _to_float_or_none(r.get("B365D"))
        b365a = _to_float_or_none(r.get("B365A"))

        rec: Dict = {
            "fecha": fecha_str,
            "hora": hora,

            # En tu BD existen 'division' y 'div' (NOT NULL)
            "division": r.get("Div"),
            "div": r.get("Div"),

            "home_team": r.get("HomeTeam"),
            "away_team": r.get("AwayTeam"),

            "b365h": b365h,
            "b365d": b365d,
            "b365a": b365a,

            "l_score": l_score,
            "h_t_score": h_t_score,
            "a_t_score": a_t_score,
            "match_score": match_score,

            "match_class": match_class,
            "pick_type": pick_type,

            # model_class NOT NULL en tu BD (por tu histórico de cambios)
            "model_class": match_class if match_class is not None and str(match_class).strip() != "" else pick_type,

            # por defecto
            "apuesta_real": "NO",

            # NUEVO: estrategia (la podrás cambiar luego en el formulario)
            "estrategia": "Convexidad",
        }

        records.append(rec)

    resp = supabase.table("seguimiento").insert(records).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en seguimiento: {resp.error}")


def fetch_seguimiento() -> pd.DataFrame:
    """Devuelve toda la tabla 'seguimiento' como DataFrame."""
    resp = supabase.table("seguimiento").select("*").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo seguimiento: {resp.error}")
    data = getattr(resp, "data", None) or []
    return pd.DataFrame(data)


def update_seguimiento_from_df(
    original_df: pd.DataFrame,
    edited_df: pd.DataFrame,
    editable_cols: List[str],
) -> int:
    """
    Actualiza 'seguimiento' comparando original_df y edited_df.
    Devuelve número de filas actualizadas.
    """
    if "id" not in original_df.columns or "id" not in edited_df.columns:
        raise ValueError("Ambos DataFrames deben tener columna 'id'.")

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

            is_new_nan = pd.isna(val_new)
            is_old_nan = pd.isna(val_old)

            if is_new_nan and is_old_nan:
                continue

            if (is_new_nan and not is_old_nan) or (not is_new_nan and is_old_nan):
                changes[col] = None if is_new_nan else val_new
            else:
                if val_new != val_old:
                    changes[col] = val_new

        if not changes:
            continue

        # Normalizar ints para minutos
        for minute_col in ["close_minute_global", "close_minute_1_1", "minuto_primer_gol"]:
            if minute_col in changes:
                changes[minute_col] = _to_int_or_none(changes[minute_col])

        # RAROC numérico si viene
        if "raroc" in changes:
            changes["raroc"] = _to_float_or_none(changes["raroc"])

        resp = supabase.table("seguimiento").update(changes).eq("id", int(row_id)).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Error actualizando id={row_id} en seguimiento: {resp.error}")

        updated_rows += 1

    return updated_rows


def update_seguimiento_row(row_id: int, changes: Dict) -> None:
    """Actualiza una sola fila por id."""
    for minute_col in ["close_minute_global", "close_minute_1_1", "minuto_primer_gol"]:
        if minute_col in changes:
            changes[minute_col] = _to_int_or_none(changes[minute_col])

    if "raroc" in changes:
        changes["raroc"] = _to_float_or_none(changes["raroc"])

    resp = supabase.table("seguimiento").update(changes).eq("id", int(row_id)).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error actualizando id={row_id} en seguimiento: {resp.error}")


def compute_and_update_pct_minuto_primer_gol(df: pd.DataFrame) -> int:
    """
    Calcula percentil del minuto_primer_gol para Ideal/Buena filtrada, y guarda
    en 'pct_minuto_primer_gol' (ya existe en tu BD según tu hilo).
    """
    if df.empty:
        return 0
    if "id" not in df.columns or "minuto_primer_gol" not in df.columns:
        return 0

    work = df.copy()
    work["minuto_primer_gol"] = pd.to_numeric(work["minuto_primer_gol"], errors="coerce")

    mask = work["minuto_primer_gol"].notna()
    if "pick_type" in work.columns:
        mask &= work["pick_type"].isin(["Ideal", "Buena filtrada"])

    subset = work[mask].copy()
    if subset.empty:
        return 0

    subset["pct"] = subset["minuto_primer_gol"].rank(method="average", pct=True)

    updated = 0
    for _, row in subset.iterrows():
        row_id = row["id"]
        pct_val = float(row["pct"])

        resp = supabase.table("seguimiento").update(
            {"pct_minuto_primer_gol": pct_val}
        ).eq("id", int(row_id)).execute()

        if getattr(resp, "error", None):
            raise RuntimeError(f"Error actualizando pct_minuto_primer_gol para id={row_id}: {resp.error}")

        updated += 1

    return updated
