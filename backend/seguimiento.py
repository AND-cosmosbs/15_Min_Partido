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

        # ----- Registro final -----
        rec: Dict = {
            "fecha": fecha_str,
            "hora": hora,
            "division": r.get("Div"),
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

            "match_class": r.get("MatchClass"),
            "pick_type": r.get("PickType"),

            # El resto (stakes, odds reales, profit...) queda en NULL
        }

        records.append(rec)

    # ----- Inserción en Supabase -----
    resp = supabase.table("seguimiento").insert(records).execute()

    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en seguimiento: {resp.error}")
