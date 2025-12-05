# backend/seguimiento.py

from typing import List, Dict
import pandas as pd

from .supabase_client import supabase


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

            # Scores del modelo
            "l_score": r.get("L_score"),
            "h_t_score": r.get("H_T_score"),
            "a_t_score": r.get("A_T_score"),

            # match_score puede venir como float → convertimos si aplica
            "match_score": int(r["MatchScore"]) if pd.notna(r.get("MatchScore")) else None,

            "match_class": r.get("MatchClass"),
            "pick_type": r.get("PickType"),

            # El resto (stakes, odds reales, profit...) queda en NULL
        }

        records.append(rec)

    # ----- Inserción en Supabase -----
    resp = supabase.table("seguimiento").insert(records).execute()

    # Si Supabase devuelve error, lo mostramos
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en seguimiento: {resp.error}")
