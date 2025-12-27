# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd

from .supabase_client import supabase


# -----------------------------
# Config
# -----------------------------

@dataclass
class VixConfig:
    lookback_pct: int = 252
    ratio_alert: float = 1.30
    ratio_ok: float = 1.25
    vix_panic: float = 0.85   # percentil P85
    vix_tension: float = 0.65 # percentil P65
    vix_calm: float = 0.25    # percentil P25

    # Guardarraíl “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_p10: float = 0.10
    guardrail_vix_floor: float = 12.5  # si VIX < 12.5 => no abrir SVIX


DEFAULT_CFG = VixConfig()


# -----------------------------
# Helpers
# -----------------------------

def _to_date(x) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(x).normalize()
    except Exception:
        return None


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# -----------------------------
# Supabase: macro events
# -----------------------------

def fetch_macro_events() -> pd.DataFrame:
    """
    Tabla esperada: macro_events(fecha, label, impacto, activo)
    """
    resp = supabase.table("macro_events").select("*").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo macro_events: {resp.error}")

    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    if "activo" in df.columns:
        df["activo"] = df["activo"].fillna(True)
    return df


def macro_tomorrow_flag(fecha: pd.Timestamp, macro_df: pd.DataFrame) -> bool:
    """
    True si hay algún evento activo mañana (fecha+1).
    """
    if macro_df is None or macro_df.empty:
        return False
    tomorrow = (fecha + pd.Timedelta(days=1)).date()
    w = macro_df.copy()
    w = w[(w.get("activo", True) == True) & (w["fecha"] == tomorrow)]
    return len(w) > 0


# -----------------------------
# Yahoo download
# -----------------------------

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de:
      ^VIX, ^VXN, VIXY, SPY
    Devuelve df con columnas: date, vix, vxn, vixy, spy
    """
    import yfinance as yf  # import aquí para no romper si no está instalado en tests

    tickers = {
        "^VIX": "vix",
        "^VXN": "vxn",
        "VIXY": "vixy",
        "SPY": "spy",
    }

    out = None

    for tkr, col in tickers.items():
        data = yf.download(tkr, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if data is None or data.empty:
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo Finance para el rango {start}..{end}")

        s = data["Close"].copy()
        s.name = col
        df = s.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        if out is None:
            out = df
        else:
            out = out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


# -----------------------------
# Señales y estado
# -----------------------------

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    df: columnas date, vix, vxn, vixy, spy
    """
    w = df.copy()
    w["vix"] = _safe_num(w["vix"])
    w["vxn"] = _safe_num(w["vxn"])
    w["vixy"] = _safe_num(w["vixy"])
    w["spy"] = _safe_num(w["spy"])

    # SPY retorno diario
    w["spy_ret"] = w["spy"].pct_change()

    # Ratio VXN/VIX + dirección
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    # VIX percentiles rolling (252)
    lb = int(cfg.lookback_pct)
    w["vix_p10"] = w["vix"].rolling(lb).quantile(0.10)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(0.25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(0.50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(0.65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(0.85)

    # VIXY MA3 vs MA10
    w["vixy_ma3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma10"] = w["vixy"].rolling(10).mean()

    w["contango_ok"] = w["vixy_ma3"] < w["vixy_ma10"]   # contango estable
    w["vxn_signal"] = (w["vxn_vix_ratio"] > cfg.ratio_alert) & (w["ratio_up"] == True)

    return w


def decide_state_row(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Dict[str, Any]:
    """
    Devuelve {estado, accion, comentario}
    Reglas base (tuyas) + guardarraíl anti “VIX demasiado bajo”.
    """
    vix = row.get("vix")
    p10 = row.get("vix_p10")
    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    ratio = row.get("vxn_vix_ratio")
    ratio_up = bool(row.get("ratio_up")) if pd.notna(row.get("ratio_up")) else False

    contango_ok = bool(row.get("contango_ok")) if pd.notna(row.get("contango_ok")) else False
    spy_ret = row.get("spy_ret")
    macro_tomorrow = bool(row.get("macro_tomorrow")) if pd.notna(row.get("macro_tomorrow")) else False

    # Si todavía no hay rolling (primeros 252 días), no operamos
    if pd.isna(p25) or pd.isna(p65) or pd.isna(p85) or pd.isna(vix):
        return {"estado": "NEUTRAL", "accion": "NO DATA", "comentario": "Insuficiente histórico para percentiles (rolling 252)."}

    # Guardarraíl: VIX demasiado bajo => no abrir SVIX
    if cfg.use_guardrail:
        too_low_by_p10 = (pd.notna(p10) and vix < p10)
        too_low_by_floor = (pd.notna(vix) and vix < cfg.guardrail_vix_floor)
        if too_low_by_p10 or too_low_by_floor:
            return {
                "estado": "NEUTRAL",
                "accion": "NO OPEN SVIX",
                "comentario": "Guardarraíl: VIX extremadamente bajo (riesgo de snapback).",
            }

    # --- SVIX (todas) ---
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_tomorrow == False)
    )
    if cond_svix:
        return {"estado": "SVIX", "accion": "OPEN/HOLD SVIX", "comentario": "Régimen calmado + contango estable + sin alerta VXN ni macro."}

    # --- UVIX (mínimo 2) ---
    uvix_cond1 = vix > p65
    uvix_cond2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uvix_cond3 = (pd.notna(row.get("vixy_ma3")) and pd.notna(row.get("vixy_ma10")) and (row.get("vixy_ma3") > row.get("vixy_ma10")))
    uvix_cond4 = (pd.notna(spy_ret) and spy_ret < -0.008)

    uvix_score = sum([bool(uvix_cond1), bool(uvix_cond2), bool(uvix_cond3), bool(uvix_cond4)])
    if uvix_score >= 2:
        return {"estado": "UVIX", "accion": "TRADE UVIX (SHORT)", "comentario": f"Señal de stress (score={uvix_score})."}

    # --- PURPLE (cerrar UVIX / preparar SVIX) ---
    # vix > P85 y ratio deja de subir y vuelve contango_ok
    cond_purple = (vix > p85) and (ratio_up == False) and contango_ok
    if cond_purple:
        return {"estado": "PREP_SVIX", "accion": "CLOSE UVIX / PREPARE SVIX", "comentario": "Pánico agotándose + contango vuelve."}

    # --- NEUTRAL por defecto ---
    return {"estado": "NEUTRAL", "accion": "NO NEW POSITION", "comentario": "Régimen mixto / transición."}


def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df_feat.copy()

    # Macro tomorrow
    macro = fetch_macro_events()
    w["macro_tomorrow"] = w["date"].apply(lambda d: macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False)

    estados = []
    acciones = []
    comentarios = []
    for _, r in w.iterrows():
        res = decide_state_row(r, cfg=cfg)
        estados.append(res["estado"])
        acciones.append(res["accion"])
        comentarios.append(res["comentario"])

    w["estado"] = estados
    w["accion"] = acciones
    w["comentario"] = comentarios
    return w


# -----------------------------
# Supabase: vix_daily
# -----------------------------

def upsert_vix_daily(df: pd.DataFrame) -> int:
    """
    Upsert por fecha en vix_daily.
    Requiere que la tabla tenga 'fecha' unique.
    """
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date
    keep_cols = [
        "fecha", "vix", "vxn", "vixy", "spy", "spy_ret",
        "vxn_vix_ratio", "vix_p10", "vix_p25", "vix_p50", "vix_p65", "vix_p85",
        "vixy_ma3", "vixy_ma10", "contango_ok", "vxn_signal",
        "macro_tomorrow", "estado", "accion", "comentario",
    ]
    w = w[[c for c in keep_cols if c in w.columns]].copy()

    # a registros
    records: List[Dict[str, Any]] = w.to_dict(orient="records")

    # upsert
    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_daily: {resp.error}")

    return len(records)


def fetch_vix_daily() -> pd.DataFrame:
    resp = supabase.table("vix_daily").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo vix_daily: {resp.error}")
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# -----------------------------
# Orders
# -----------------------------

def fetch_vix_orders() -> pd.DataFrame:
    resp = supabase.table("vix_orders").select("*").order("fecha", desc=True).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo vix_orders: {resp.error}")
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


def insert_vix_order(
    fecha,
    estado: str,
    ticker: str,
    side: str,
    qty: float,
    price: Optional[float] = None,
    status: str = "PLANNED",
    notes: Optional[str] = None,
) -> None:
    rec = {
        "fecha": str(fecha),
        "estado": estado,
        "ticker": ticker,
        "side": side,
        "qty": float(qty),
        "price": float(price) if price is not None else None,
        "status": status,
        "notes": notes,
    }
    resp = supabase.table("vix_orders").insert(rec).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error insertando vix_order: {resp.error}")


# -----------------------------
# One-shot pipeline
# -----------------------------

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)
    upsert_vix_daily(out)
    return out
