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
        fecha = None
        if "Date" in r and pd.notna(r["Date"]):
            try:
                fecha = r["Date"].date()
            except Exception:
                fecha = None

        hora = None
        if "Time" in r and pd.notna(r["Time"]):
            hora = str(r["Time"])

        rec: Dict = {
            "fecha": fecha,
            "hora": hora,
            "division": r.get("Div"),
            "home_team": r.get("HomeTeam"),
            "away_team": r.get("AwayTeam"),
            "b365h": r.get("B365H"),
            "b365d": r.get("B365D"),
            "b365a": r.get("B365A"),
            "l_score": r.get("L_score"),
            "h_t_score": r.get("H_T_score"),
            "a_t_score": r.get("A_T_score"),
            "match_score": int(r["MatchScore"]) if pd.notna(r.get("MatchScore")) else None,
            "match_class": r.get("MatchClass"),
            "pick_type": r.get("PickType"),
        }

        records.append(rec)

    resp = supabase.table("seguimiento").insert(records).execute()

    # Manejo simple de error
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en seguimiento: {resp.error}")
