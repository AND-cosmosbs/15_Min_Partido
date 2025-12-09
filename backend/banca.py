# backend/banca.py

from typing import Optional
import pandas as pd

from .supabase_client import supabase


# ---------------------------------------------------------
# LECTURA / INSERCIÓN DE MOVIMIENTOS DE BANCA
# ---------------------------------------------------------

def fetch_banca_movimientos() -> pd.DataFrame:
    """
    Devuelve la tabla banca_movimientos como DataFrame,
    ordenada por fecha ascendente.
    """
    resp = supabase.table("banca_movimientos").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo banca_movimientos: {resp.error}")

    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)

    if not df.empty:
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        if "importe" in df.columns:
            df["importe"] = pd.to_numeric(df["importe"], errors="coerce")

    return df


def insert_banca_movimiento(fecha, tipo: str, importe: float, comentario: Optional[str] = None) -> None:
    """
    Inserta un movimiento de banca:
      - tipo: 'DEPOSITO', 'RETIRADA', 'AJUSTE'
    """
    if fecha is None:
        raise ValueError("La fecha es obligatoria para un movimiento de banca.")

    # fecha puede venir como date/datetime → lo pasamos a string ISO
    fecha_str = pd.to_datetime(fecha).date().isoformat()

    row = {
        "fecha": fecha_str,
        "tipo": tipo,
        "importe": float(importe),
        "comentario": comentario,
    }

    resp = supabase.table("banca_movimientos").insert(row).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en banca_movimientos: {resp.error}")


# ---------------------------------------------------------
# SERIE DE BANCA + ROI ACUMULADO
# ---------------------------------------------------------

def compute_banca_series(
    banca_inicial: float,
    mov_df: pd.DataFrame,
    bets_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una serie temporal con:
      - movimientos de banca (depósitos, retiradas, ajustes)
      - beneficios de apuestas (profit_euros)
    Devuelve un DataFrame con:
      fecha, mov_dia, mov_acum, profit_dia, profit_acum,
      capital_base_acum (banca_inicial + mov_acum),
      banca_actual (capital_base_acum + profit_acum),
      roi_sobre_inicial, roi_sobre_capital_acum
    """

    # --- Movimientos de banca ---
    if mov_df is not None and not mov_df.empty:
        df_mov = mov_df.copy()
        df_mov["fecha"] = pd.to_datetime(df_mov["fecha"], errors="coerce")
        df_mov = df_mov[df_mov["fecha"].notna()]
        df_mov["importe"] = pd.to_numeric(df_mov["importe"], errors="coerce").fillna(0)

        def _signed_mov(row):
            t = (row.get("tipo") or "").upper()
            if t == "DEPOSITO":
                return row["importe"]
            elif t == "RETIRADA":
                return -row["importe"]
            # AJUSTE u otros: se toma tal cual
            return row["importe"]

        df_mov["mov_signed"] = df_mov.apply(_signed_mov, axis=1)
        mov_by_date = df_mov.groupby("fecha")["mov_signed"].sum()
    else:
        mov_by_date = pd.Series(dtype=float)

    # --- Apuestas (profit por día) ---
    if bets_df is not None and not bets_df.empty:
        df_bet = bets_df.copy()
        if "fecha" in df_bet.columns:
            df_bet["fecha"] = pd.to_datetime(df_bet["fecha"], errors="coerce")
            df_bet = df_bet[df_bet["fecha"].notna()]
        else:
            # Si no hay fecha, no se puede construir serie
            return pd.DataFrame()

        df_bet["profit_euros"] = pd.to_numeric(df_bet.get("profit_euros", 0), errors="coerce").fillna(0)
        bets_by_date = df_bet.groupby("fecha")["profit_euros"].sum()
    else:
        bets_by_date = pd.Series(dtype=float)

    # --- Unimos fechas ---
    all_dates = sorted(set(mov_by_date.index).union(set(bets_by_date.index)))
    if not all_dates:
        return pd.DataFrame()

    rows = []
    cum_mov = 0.0
    cum_profit = 0.0

    for d in all_dates:
        mov_dia = float(mov_by_date.get(d, 0.0))
        prof_dia = float(bets_by_date.get(d, 0.0))

        cum_mov += mov_dia
        cum_profit += prof_dia

        capital_base_acum = banca_inicial + cum_mov      # capital aportado neto (inicial + dep - ret)
        banca_actual = capital_base_acum + cum_profit    # capital + beneficios acumulados

        roi_sobre_inicial = (
            cum_profit / banca_inicial if banca_inicial > 0 else None
        )
        roi_sobre_capital_acum = (
            cum_profit / capital_base_acum if capital_base_acum > 0 else None
        )

        rows.append({
            "fecha": d,
            "mov_dia": mov_dia,
            "mov_acum": cum_mov,
            "profit_dia": prof_dia,
            "profit_acum": cum_profit,
            "capital_base_acum": capital_base_acum,
            "banca_actual": banca_actual,
            "roi_sobre_inicial": roi_sobre_inicial,
            "roi_sobre_capital_acum": roi_sobre_capital_acum,
        })

    return pd.DataFrame(rows)
