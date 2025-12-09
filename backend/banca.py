# backend/banca.py

from typing import Optional
import pandas as pd

from .supabase_client import supabase


def fetch_banca_movimientos() -> pd.DataFrame:
    """
    Lee todos los movimientos de banca desde la tabla 'banca_movimientos'
    y los devuelve como DataFrame ordenados por fecha.
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


def insert_banca_movimiento(
    fecha,
    tipo: str,
    importe: float,
    comentario: Optional[str] = None,
) -> None:
    """
    Inserta un movimiento en la tabla 'banca_movimientos'.

    tipo: 'DEPOSITO', 'RETIRADA', 'AJUSTE'
    fecha: date o string 'YYYY-MM-DD'
    importe: número positivo (el signo se gestiona por tipo)
    """
    if not tipo:
        raise ValueError("tipo es obligatorio (DEPOSITO / RETIRADA / AJUSTE)")

    rec = {
        "fecha": str(fecha),
        "tipo": tipo,
        "importe": float(importe),
        "comentario": comentario,
    }

    resp = supabase.table("banca_movimientos").insert(rec).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en banca_movimientos: {resp.error}")
