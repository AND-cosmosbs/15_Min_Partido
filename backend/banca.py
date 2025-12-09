# backend/banca.py

import pandas as pd
from typing import Optional

from .supabase_client import supabase


# --------------------------------------------------
# Leer movimientos de banca
# --------------------------------------------------

def fetch_banca_movimientos() -> pd.DataFrame:
    resp = (
        supabase
        .table("banca_movimientos")
        .select("*")
        .order("fecha", desc=False)
        .order("created_at", desc=False)
        .execute()
    )

    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)

    df = pd.DataFrame(resp.data or [])

    if df.empty:
        return df

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["importe"] = pd.to_numeric(df["importe"], errors="coerce")

    return df


# --------------------------------------------------
# Insertar movimiento
# --------------------------------------------------

def insert_banca_movimiento(
    fecha,
    tipo: str,
    importe: float,
    comentario: Optional[str] = None,
):
    rec = {
        "fecha": pd.to_datetime(fecha).date().isoformat(),
        "tipo": tipo,
        "importe": float(importe),
        "comentario": comentario or "",
    }

    resp = supabase.table("banca_movimientos").insert(rec).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)


# --------------------------------------------------
# Calcular curva de banca
# --------------------------------------------------

def compute_banca_timeseries(
    apuestas_df: pd.DataFrame,
    mov_df: pd.DataFrame,
    banca_inicial: float,
) -> pd.DataFrame:
    """
    Construye serie diaria:
    - beneficios de apuestas reales
    - movimientos de banca
    - banca acumulada
    - ROI sobre banca inicial
    """

    # --- apuestas (solo reales)
    ap = apuestas_df.copy()
    ap["fecha"] = pd.to_datetime(ap["fecha"], errors="coerce")

    ap = ap[ap["apuesta_real"] == "SI"]

    ap["profit_euros"] = pd.to_numeric(ap["profit_euros"], errors="coerce")

    profit_dia = (
        ap.dropna(subset=["fecha"])
        .groupby(ap["fecha"].dt.date)["profit_euros"]
        .sum()
    )

    # --- movimientos banca
    if mov_df.empty:
        mov_dia = pd.Series(dtype="float64")
    else:
        mov_df = mov_df.copy()
        mov_df["fecha"] = pd.to_datetime(mov_df["fecha"], errors="coerce")
        mov_dia = (
            mov_df.dropna(subset=["fecha"])
            .groupby(mov_df["fecha"].dt.date)["importe"]
            .sum()
        )

    # --- fechas combinadas
    fechas = sorted(set(profit_dia.index) | set(mov_dia.index))

    rows = []
    banca = float(banca_inicial)

    for f in fechas:
        profit = float(profit_dia.get(f, 0.0))
        mov = float(mov_dia.get(f, 0.0))

        banca_prev = banca
        banca = banca + profit + mov

        roi_acum = (banca / banca_inicial - 1) if banca_inicial > 0 else None

        rows.append({
            "fecha": pd.to_datetime(f),
            "profit_dia": profit,
            "mov_banca": mov,
            "banca": banca,
            "roi_banca": roi_acum,
        })

    return pd.DataFrame(rows)
