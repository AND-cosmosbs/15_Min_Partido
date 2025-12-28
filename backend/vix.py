# backend/vix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd

from .supabase_client import supabase


# =========================================================
# CONFIG
# =========================================================

@dataclass
class VixConfig:
    lookback_pct: int = 252
    ratio_alert: float = 1.30
    ratio_ok: float = 1.25

    # percentiles
    p25: float = 0.25
    p50: float = 0.50
    p65: float = 0.65
    p85: float = 0.85

    # guardrail anti “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_vix_floor: float = 12.5


DEFAULT_CFG = VixConfig()


# =========================================================
# HELPERS
# =========================================================

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _as_date_str(x) -> Optional[str]:
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return None
        return d.date().isoformat()
    except Exception:
        return None


def _try_fetch_macro_events() -> pd.DataFrame:
    """
    Si existe la tabla macro_events la usamos. Si no existe, devolvemos vacío (NO rompe).
    """
    try:
        resp = supabase.table("macro_events").select("*").execute()
        if getattr(resp, "error", None):
            return pd.DataFrame()
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
    except Exception:
        return pd.DataFrame()


def _macro_tomorrow_flag(fecha: pd.Timestamp, macro_df: pd.DataFrame) -> bool:
    if macro_df is None or macro_df.empty:
        return False
    tomorrow = (fecha + pd.Timedelta(days=1)).date()
    w = macro_df.copy()
    w = w[(w["activo"] == True) & (w["fecha"] == tomorrow)]
    return len(w) > 0


# =========================================================
# YAHOO DOWNLOAD
# =========================================================

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de:
      ^VIX, ^VXN, VIXY, SPY
    Devuelve df con columnas: date, vix, vxn, vixy, spy
    """
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
            raise RuntimeError(f"No hay datos Yahoo para {tkr} en {start}..{end}")

        # yfinance a veces devuelve columnas con espacios raros / multiindex en algunos casos:
        if isinstance(data.columns, pd.MultiIndex):
            # intentamos quedarnos con Close
            if ("Close" in data.columns.get_level_values(0)):
                close = data["Close"]
                if isinstance(close, pd.DataFrame) and close.shape[1] >= 1:
                    s = close.iloc[:, 0].copy()
                else:
                    s = close.copy()
            else:
                # fallback
                s = data.iloc[:, 0].copy()
        else:
            if "Close" in data.columns:
                s = data["Close"].copy()
            else:
                s = data.iloc[:, 0].copy()

        s.name = col
        df = s.reset_index()
        # Date o Datetime según yfinance
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "date"}, inplace=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        out = df if out is None else out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


# =========================================================
# FEATURES + REGLAS (adaptadas a tus tablas SQL)
# =========================================================

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    df: columnas date, vix, vxn, vixy, spy
    """
    w = df.copy()

    # Asegura columnas (y así si algo falla, lo verás claro)
    for c in ["date", "vix", "vxn", "vixy", "spy"]:
        if c not in w.columns:
            raise KeyError(c)

    w["vix"] = _safe_num(w["vix"])
    w["vxn"] = _safe_num(w["vxn"])
    w["vixy"] = _safe_num(w["vixy"])
    w["spy"] = _safe_num(w["spy"])

    w["spy_return"] = w["spy"].pct_change()
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    lb = int(cfg.lookback_pct)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(cfg.p25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(cfg.p50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(cfg.p65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(cfg.p85)

    w["vixy_ma_3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma_10"] = w["vixy"].rolling(10).mean()

    return w


def _vix_regime(row: pd.Series) -> str:
    vix = row.get("vix")
    p25 = row.get("vix_p25")
    p50 = row.get("vix_p50")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")
    if pd.isna(vix) or pd.isna(p25) or pd.isna(p50) or pd.isna(p65) or pd.isna(p85):
        return "NA"
    if vix < p25:
        return "CALMA"
    if vix < p65:
        return "ALERTA"  # zona media amplia
    if vix < p85:
        return "TENSION"
    return "PANICO"


def _contango_estado(row: pd.Series) -> str:
    ma3 = row.get("vixy_ma_3")
    ma10 = row.get("vixy_ma_10")
    if pd.isna(ma3) or pd.isna(ma10):
        return "NA"
    if ma3 < ma10:
        return "CONTANGO"
    if ma3 > ma10:
        return "BACKWARDATION"
    return "TRANSICION"


def decide_state_row(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Dict[str, Any]:
    """
    Devuelve estado + motivo, usando:
    - vix percentiles
    - ratio vxn/vix
    - contango proxy (VIXY MA3 vs MA10)
    - spy_return
    - macro_evento (si existe macro_events)
    """
    vix = row.get("vix")
    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    ratio = row.get("vxn_vix_ratio")
    ratio_up = bool(row.get("ratio_up")) if pd.notna(row.get("ratio_up")) else False

    ma3 = row.get("vixy_ma_3")
    ma10 = row.get("vixy_ma_10")
    contango_ok = (pd.notna(ma3) and pd.notna(ma10) and (ma3 < ma10))
    contango_bad = (pd.notna(ma3) and pd.notna(ma10) and (ma3 > ma10))

    spy_ret = row.get("spy_return")
    macro_evento = bool(row.get("macro_evento")) if pd.notna(row.get("macro_evento")) else False

    # no hay percentiles aún
    if pd.isna(vix) or pd.isna(p25) or pd.isna(p65) or pd.isna(p85):
        return {"estado": "NEUTRAL", "motivo": "Insuficiente histórico rolling 252."}

    # guardrail: vix demasiado bajo
    if cfg.use_guardrail and pd.notna(vix) and vix < cfg.guardrail_vix_floor:
        return {"estado": "NEUTRAL", "motivo": f"Guardrail: VIX < {cfg.guardrail_vix_floor} (snapback risk)."}

    # SVIX (todas)
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_evento is False)
    )
    if cond_svix:
        return {"estado": "SVIX", "motivo": "VIX < P25 + ratio OK + contango + sin macro."}

    # UVIX (2 de 4)
    c1 = vix > p65
    c2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    c3 = contango_bad
    c4 = (pd.notna(spy_ret) and spy_ret < -0.008)

    score = sum([bool(c1), bool(c2), bool(c3), bool(c4)])
    if score >= 2:
        return {"estado": "UVIX", "motivo": f"Stress score={score} (VIX/radio/contango/SPY)."}

    # PURPLE (tu regla “pánico agotándose”)
    cond_purple = (vix > p85) and (ratio_up is False) and contango_ok
    if cond_purple:
        return {"estado": "CERRAR_UVIX", "motivo": "Pánico + ratio deja de subir + contango vuelve."}

    return {"estado": "NEUTRAL", "motivo": "Régimen mixto / transición."}


def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df_feat.copy()

    macro = _try_fetch_macro_events()
    w["macro_evento"] = w["date"].apply(
        lambda d: _macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False
    )

    estados = []
    motivos = []
    for _, r in w.iterrows():
        res = decide_state_row(r, cfg=cfg)
        estados.append(res["estado"])
        motivos.append(res["motivo"])

    w["estado"] = estados
    w["motivo"] = motivos
    w["vix_regime"] = w.apply(_vix_regime, axis=1)
    w["contango_estado"] = w.apply(_contango_estado, axis=1)

    return w


# =========================================================
# SUPABASE I/O (AJUSTADO A TUS TABLAS SQL)
# =========================================================

def upsert_vix_daily(df: pd.DataFrame) -> int:
    """
    Escribe en tu tabla vix_daily (la del SQL que pegaste):
      fecha PK
      vix, vxn_vix_ratio, vix_p25/p50/p65/p85, vix_regime,
      vixy_ma_3, vixy_ma_10, contango_estado
    """
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = w["date"].apply(_as_date_str)

    records: List[Dict[str, Any]] = []
    for _, r in w.iterrows():
        if not r.get("fecha"):
            continue
        rec = {
            "fecha": r.get("fecha"),
            "vix": None if pd.isna(r.get("vix")) else float(r.get("vix")),
            "vxn_vix_ratio": None if pd.isna(r.get("vxn_vix_ratio")) else float(r.get("vxn_vix_ratio")),
            "vix_p25": None if pd.isna(r.get("vix_p25")) else float(r.get("vix_p25")),
            "vix_p50": None if pd.isna(r.get("vix_p50")) else float(r.get("vix_p50")),
            "vix_p65": None if pd.isna(r.get("vix_p65")) else float(r.get("vix_p65")),
            "vix_p85": None if pd.isna(r.get("vix_p85")) else float(r.get("vix_p85")),
            "vix_regime": r.get("vix_regime"),
            "vixy_ma_3": None if pd.isna(r.get("vixy_ma_3")) else float(r.get("vixy_ma_3")),
            "vixy_ma_10": None if pd.isna(r.get("vixy_ma_10")) else float(r.get("vixy_ma_10")),
            "contango_estado": r.get("contango_estado"),
        }
        records.append(rec)

    if not records:
        return 0

    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_daily: {resp.error}")

    return len(records)


def upsert_vix_signal(df: pd.DataFrame) -> int:
    """
    Escribe 1 fila por día en tu tabla vix_signal (fecha PK).
    """
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = w["date"].apply(_as_date_str)

    records: List[Dict[str, Any]] = []
    for _, r in w.iterrows():
        if not r.get("fecha"):
            continue
        rec = {
            "fecha": r.get("fecha"),
            "estado": r.get("estado") or "NEUTRAL",
            "motivo": r.get("motivo"),
            "vix": None if pd.isna(r.get("vix")) else float(r.get("vix")),
            "vxn_vix_ratio": None if pd.isna(r.get("vxn_vix_ratio")) else float(r.get("vxn_vix_ratio")),
            "contango_estado": r.get("contango_estado"),
            "spy_return": None if pd.isna(r.get("spy_return")) else float(r.get("spy_return")),
            "macro_evento": bool(r.get("macro_evento")) if pd.notna(r.get("macro_evento")) else False,
        }
        records.append(rec)

    if not records:
        return 0

    resp = supabase.table("vix_signal").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_signal: {resp.error}")

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


def fetch_vix_signal() -> pd.DataFrame:
    resp = supabase.table("vix_signal").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo vix_signal: {resp.error}")
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# =========================================================
# ONE-SHOT PIPELINE
# =========================================================

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)
    upsert_vix_daily(out)
    upsert_vix_signal(out)
    return out
