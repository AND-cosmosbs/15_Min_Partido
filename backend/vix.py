# backend/vix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd
from .supabase_client import supabase


@dataclass
class VixConfig:
    lookback_pct: int = 252
    ratio_alert: float = 1.30
    ratio_ok: float = 1.25
    use_guardrail: bool = True
    guardrail_p10: float = 0.10
    guardrail_vix_floor: float = 12.5


DEFAULT_CFG = VixConfig()


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fetch_macro_events() -> pd.DataFrame:
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
    else:
        df["activo"] = True
    return df


def macro_tomorrow_flag(fecha: pd.Timestamp, macro_df: pd.DataFrame) -> bool:
    if macro_df is None or macro_df.empty:
        return False
    tomorrow = (pd.to_datetime(fecha).normalize() + pd.Timedelta(days=1)).date()
    w = macro_df.copy()
    w = w[(w.get("activo", True) == True) & (w["fecha"] == tomorrow)]
    return len(w) > 0


def _extract_close(df: pd.DataFrame) -> pd.Series:
    """
    yfinance puede devolver columnas normales o MultiIndex.
    Priorizamos 'Close', si no, 'Adj Close'.
    """
    if df is None or df.empty:
        raise RuntimeError("DataFrame vacío desde yfinance")

    cols = df.columns
    # MultiIndex -> buscamos nivel 0
    if isinstance(cols, pd.MultiIndex):
        # intentamos ("Close", ...) o ("Adj Close", ...)
        for name in ["Close", "Adj Close"]:
            matches = [c for c in cols if c[0] == name]
            if matches:
                return df[matches[0]].copy()
        # fallback: primera columna
        return df[cols[0]].copy()

    # columnas normales
    if "Close" in df.columns:
        return df["Close"].copy()
    if "Adj Close" in df.columns:
        return df["Adj Close"].copy()

    # fallback
    return df.iloc[:, 0].copy()


def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    tickers = {
        "^VIX": "vix",
        "^VXN": "vxn",
        "VIXY": "vixy",
        "SPY": "spy",
    }

    out = None

    for tkr, col in tickers.items():
        data = yf.download(
            tkr,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if data is None or data.empty:
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo Finance ({start}..{end}).")

        s = _extract_close(data)
        s.name = col

        df = s.reset_index()

        # yfinance puede devolver Date o Datetime
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "date"}, inplace=True)

        if "date" not in df.columns:
            raise RuntimeError(f"No se pudo identificar columna fecha para {tkr}. Columnas: {list(df.columns)}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        if out is None:
            out = df
        else:
            out = out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)

    # blindaje: asegurar columnas esperadas
    for c in ["vix", "vxn", "vixy", "spy"]:
        if c not in out.columns:
            out[c] = pd.NA

    return out


def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()

    for c in ["vix", "vxn", "vixy", "spy"]:
        w[c] = _safe_num(w[c])

    w["spy_ret"] = w["spy"].pct_change()

    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    lb = int(cfg.lookback_pct)
    w["vix_p10"] = w["vix"].rolling(lb).quantile(0.10)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(0.25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(0.50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(0.65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(0.85)

    w["vixy_ma3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma10"] = w["vixy"].rolling(10).mean()

    w["contango_ok"] = w["vixy_ma3"] < w["vixy_ma10"]
    w["vxn_signal"] = (w["vxn_vix_ratio"] > cfg.ratio_alert) & (w["ratio_up"] == True)

    return w


def decide_state_row(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Dict[str, Any]:
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

    if pd.isna(p25) or pd.isna(p65) or pd.isna(p85) or pd.isna(vix):
        return {"estado": "NEUTRAL", "accion": "NO DATA", "comentario": "Insuficiente histórico para rolling 252."}

    if cfg.use_guardrail:
        too_low_by_p10 = (pd.notna(p10) and vix < p10)
        too_low_by_floor = (pd.notna(vix) and vix < cfg.guardrail_vix_floor)
        if too_low_by_p10 or too_low_by_floor:
            return {
                "estado": "NEUTRAL",
                "accion": "NO OPEN SVIX",
                "comentario": "Guardarraíl: VIX extremadamente bajo (snapback).",
            }

    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_tomorrow == False)
    )
    if cond_svix:
        return {"estado": "SVIX", "accion": "OPEN/HOLD SVIX", "comentario": "Calma + contango + sin macro mañana."}

    uvix_cond1 = vix > p65
    uvix_cond2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uvix_cond3 = bool(row.get("vixy_ma3") > row.get("vixy_ma10")) if pd.notna(row.get("vixy_ma3")) and pd.notna(row.get("vixy_ma10")) else False
    uvix_cond4 = (pd.notna(spy_ret) and spy_ret < -0.008)

    uvix_score = sum([bool(uvix_cond1), bool(uvix_cond2), bool(uvix_cond3), bool(uvix_cond4)])
    if uvix_score >= 2:
        return {"estado": "UVIX", "accion": "TRADE UVIX (SHORT)", "comentario": f"Stress score={uvix_score}."}

    cond_purple = (vix > p85) and (ratio_up == False) and contango_ok
    if cond_purple:
        return {"estado": "PREP_SVIX", "accion": "CLOSE UVIX / PREPARE SVIX", "comentario": "Pánico agotándose + contango vuelve."}

    return {"estado": "NEUTRAL", "accion": "NO NEW POSITION", "comentario": "Transición."}


def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df_feat.copy()
    macro = fetch_macro_events()
    w["macro_tomorrow"] = w["date"].apply(lambda d: macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False)

    estados, acciones, comentarios = [], [], []
    for _, r in w.iterrows():
        res = decide_state_row(r, cfg=cfg)
        estados.append(res["estado"])
        acciones.append(res["accion"])
        comentarios.append(res["comentario"])

    w["estado"] = estados
    w["accion"] = acciones
    w["comentario"] = comentarios
    return w


def upsert_vix_daily(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date

    # Guardamos también en tus columnas antiguas (vixy_ma_3 / vixy_ma_10) si existen
    if "vixy_ma3" in w.columns:
        w["vixy_ma_3"] = w["vixy_ma3"]
    if "vixy_ma10" in w.columns:
        w["vixy_ma_10"] = w["vixy_ma10"]

    keep_cols = [
        "fecha",
        "vix", "vxn", "vixy", "spy",
        "spy_ret",
        "vxn_vix_ratio",
        "vix_p10", "vix_p25", "vix_p50", "vix_p65", "vix_p85",
        "vixy_ma3", "vixy_ma10", "vixy_ma_3", "vixy_ma_10",
        "contango_ok", "vxn_signal", "macro_tomorrow",
        "estado", "accion", "comentario",
    ]
    w = w[[c for c in keep_cols if c in w.columns]].copy()

    records: List[Dict[str, Any]] = w.to_dict(orient="records")
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


def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)
    upsert_vix_daily(out)
    return out
