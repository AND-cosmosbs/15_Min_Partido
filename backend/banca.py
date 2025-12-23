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

    Reglas:
    - DEPOSITO: importe debe ser > 0
    - RETIRADA: importe debe ser > 0
    - AJUSTE: importe puede ser positivo o negativo (para corregir la banca)
    """
    if not tipo:
        raise ValueError("tipo es obligatorio (DEPOSITO / RETIRADA / AJUSTE)")

    tipo = str(tipo).strip().upper()
    if tipo not in {"DEPOSITO", "RETIRADA", "AJUSTE"}:
        raise ValueError("tipo inválido. Usa: DEPOSITO / RETIRADA / AJUSTE")

    try:
        imp = float(importe)
    except Exception:
        raise ValueError("importe debe ser numérico")

    # ✅ Permitimos negativo SOLO en AJUSTE
    if tipo in {"DEPOSITO", "RETIRADA"}:
        if imp <= 0:
            raise ValueError("En DEPOSITO/RETIRADA el importe debe ser > 0")
    else:  # AJUSTE
        if imp == 0:
            raise ValueError("En AJUSTE, el importe no puede ser 0 (usa + o - para ajustar)")

    rec = {
        "fecha": str(fecha),
        "tipo": tipo,
        "importe": imp,  # tal cual; en AJUSTE puede ser negativo
        "comentario": comentario,
    }

    resp = supabase.table("banca_movimientos").insert(rec).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando en banca_movimientos: {resp.error}")
