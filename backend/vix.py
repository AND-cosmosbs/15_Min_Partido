# backend/vix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd

from .supabase_client import supabase


# =============================
# Config
# =============================

@dataclass
class VixConfig:
    lookback_pct: int = 252

    ratio_alert: float = 1.30
    ratio_ok: float = 1.25

    # percentiles “régimen”
    vix_panic_q: float = 0.85   # P85
    vix_tension_q: float = 0.65 # P65
    vix_calm_q: float = 0.25    # P25

    # Guardarraíl anti “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_p10_q: float = 0.10
    guardrail_vix_floor: float = 12.5  # si VIX < 12.5 => NO abrir SVIX


DEFAULT_CFG = VixConfig()


# =============================
# Helpers: robust + JSON-safe
# =============================

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _json_safe_value(x):
    """Convierte a tipos JSON-safe: None/float/int/str/bool."""
    if x is None:
        return None
    # pd.NA / NaN / NaT
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # fechas
    if isinstance(x, (pd.Timestamp,)):
        # fecha normalizada (YYYY-MM-DD)
        return x.date().isoformat()

    # date/datetime nativos
    import datetime as _dt
    if isinstance(x, (_dt.date, _dt.datetime)):
        # si es datetime -> date, si es date -> iso
        d = x.date() if isinstance(x, _dt.datetime) else x
        return d.isoformat()

    # numpy types -> python types
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    # bool/str/int/float
    if isinstance(x, (bool, int, float, str)):
        return x

    # fallback
    return str(x)


def _df_to_records_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convierte df -> records con valores JSON-safe (sin pd.NA, sin date no-serializable)."""
    if df is None or df.empty:
        return []
    out = []
    for rec in df.to_dict(orient="records"):
        out.append({k: _json_safe_value(v) for k, v in rec.items()})
    return out


# =============================
# Macro events (si existe tabla)
# =============================

def fetch_macro_events() -> pd.DataFrame:
    """
    Tabla esperada: macro_events(fecha, label, impacto, activo)
    Si no existe o está vacía, devuelve DF vacío.
    """
    resp = supabase.table("macro_events").select("*").execute()
    if getattr(resp, "error", None):
        # si no existe, no reventamos el sistema VIX
        return pd.DataFrame()

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
    if macro_df is None or macro_df.empty:
        return False
    tomorrow = (pd.to_datetime(fecha) + pd.Timedelta(days=1)).date()
    w = macro_df.copy()
    if "activo" in w.columns:
        w = w[w["activo"] == True]
    if "fecha" not in w.columns:
        return False
    w = w[w["fecha"] == tomorrow]
    return len(w) > 0


# =============================
# Yahoo download (robusto)
# =============================

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de: ^VIX, ^VXN, VIXY, SPY
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
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo Finance ({start}..{end}).")

        # Close robusto
        if "Close" not in data.columns:
            # por si cambia el nombre en algún caso raro
            close_col = [c for c in data.columns if str(c).lower() == "close"]
            if not close_col:
                raise RuntimeError(f"{tkr}: no encuentro columna Close.")
            s = data[close_col[0]].copy()
        else:
            s = data["Close"].copy()

        s.name = col
        df = s.reset_index()

        # columna fecha robusta
        # (puede llamarse Date, Datetime, etc.)
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "date"}, inplace=True)
        else:
            # primera columna = fecha
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        out = df if out is None else out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


# =============================
# Features + estado
# =============================

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()

    # asegurar columnas esperadas
    for c in ["vix", "vxn", "vixy", "spy"]:
        if c not in w.columns:
            raise KeyError(c)

    w["vix"] = _safe_num(w["vix"])
    w["vxn"] = _safe_num(w["vxn"])
    w["vixy"] = _safe_num(w["vixy"])
    w["spy"] = _safe_num(w["spy"])

    # SPY retorno diario
    w["spy_ret"] = w["spy"].pct_change()

    # Ratio VXN/VIX + dirección
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    # percentiles rolling
    lb = int(cfg.lookback_pct)
    w["vix_p10"] = w["vix"].rolling(lb).quantile(cfg.guardrail_p10_q)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(cfg.vix_calm_q)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(0.50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(cfg.vix_tension_q)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(cfg.vix_panic_q)

    # VIXY MA3 vs MA10
    w["vixy_ma3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma10"] = w["vixy"].rolling(10).mean()

    # contango proxy
    w["contango_ok"] = w["vixy_ma3"] < w["vixy_ma10"]
    return w


def _vix_regime(row: pd.Series) -> Optional[str]:
    vix = row.get("vix")
    p25 = row.get("vix_p25")
    p50 = row.get("vix_p50")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")
    if pd.isna(vix) or pd.isna(p25) or pd.isna(p50) or pd.isna(p65) or pd.isna(p85):
        return None
    if vix < p25:
        return "CALMA"
    if vix < p50:
        return "ALERTA"
    if vix < p65:
        return "ALERTA"
    if vix < p85:
        return "TENSION"
    return "PANICO"


def _contango_estado(row: pd.Series) -> Optional[str]:
    ma3 = row.get("vixy_ma3")
    ma10 = row.get("vixy_ma10")
    if pd.isna(ma3) or pd.isna(ma10):
        return None
    if ma3 < ma10:
        return "CONTANGO"
    # transición si están muy cerca (tolerancia 0.1%)
    if ma10 != 0 and abs((ma3 - ma10) / ma10) <= 0.001:
        return "TRANSICION"
    return "BACKWARDATION"


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

    # sin rolling aún
    if pd.isna(p25) or pd.isna(p65) or pd.isna(p85) or pd.isna(vix):
        return {"estado": "NEUTRAL", "motivo": "NO DATA (rolling 252 insuficiente)"}

    # guardarraíl
    if cfg.use_guardrail:
        too_low_by_p10 = (pd.notna(p10) and vix < p10)
        too_low_by_floor = (pd.notna(vix) and vix < cfg.guardrail_vix_floor)
        if too_low_by_p10 or too_low_by_floor:
            return {"estado": "NEUTRAL", "motivo": "Guardarraíl: VIX extremadamente bajo"}

    # SVIX
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_tomorrow is False)
    )
    if cond_svix:
        return {"estado": "SVIX", "motivo": "CALMA + contango estable + VXN sin alerta + sin macro mañana"}

    # UVIX: score>=2
    uv1 = vix > p65
    uv2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uv3 = (pd.notna(row.get("vixy_ma3")) and pd.notna(row.get("vixy_ma10")) and (row.get("vixy_ma3") > row.get("vixy_ma10")))
    uv4 = (pd.notna(spy_ret) and spy_ret < -0.008)
    score = sum([bool(uv1), bool(uv2), bool(uv3), bool(uv4)])
    if score >= 2:
        return {"estado": "UVIX", "motivo": f"Stress score={score}"}

    # CERRAR_UVIX / preparar SVIX
    cond_purple = (vix > p85) and (ratio_up is False) and contango_ok
    if cond_purple:
        return {"estado": "CERRAR_UVIX", "motivo": "PANICO agotándose + contango vuelve"}

    return {"estado": "NEUTRAL", "motivo": "Régimen mixto / transición"}


def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df_feat.copy()

    macro = fetch_macro_events()
    w["macro_tomorrow"] = w["date"].apply(
        lambda d: macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False
    )

    estados = []
    motivos = []
    for _, r in w.iterrows():
        res = decide_state_row(r, cfg=cfg)
        estados.append(res["estado"])
        motivos.append(res["motivo"])

    w["estado"] = estados
    w["motivo"] = motivos

    # etiquetas compatibles con tu SQL
    w["vix_regime"] = w.apply(_vix_regime, axis=1)
    w["contango_estado"] = w.apply(_contango_estado, axis=1)

    return w


# =============================
# Supabase upserts (tus tablas)
# =============================

def upsert_vix_daily(df: pd.DataFrame) -> int:
    """
    Upsert por fecha en vix_daily.
    Limpia pd.NA/NaN y convierte fechas a str para JSON.
    """
    if df is None or df.empty:
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

    # ---- limpieza JSON-safe ----
    # fechas a string
    w["fecha"] = w["fecha"].apply(lambda x: x.isoformat() if pd.notna(x) else None)

    # NaN/pd.NA -> None
    w = w.where(pd.notna(w), None)

    records = w.to_dict(orient="records")

    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_daily: {resp.error}")

    return len(records)


def upsert_vix_signal(df: pd.DataFrame) -> int:
    """
    Upsert en vix_signal (tu esquema).
    Columns usadas:
      fecha, estado, motivo, vix, vxn_vix_ratio, contango_estado, spy_return, macro_evento
    """
    if df is None or df.empty:
        return 0

    w = df.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date

    payload = pd.DataFrame({
        "fecha": w["fecha"],
        "estado": w.get("estado"),
        "motivo": w.get("motivo"),
        "vix": w.get("vix"),
        "vxn_vix_ratio": w.get("vxn_vix_ratio"),
        "contango_estado": w.get("contango_estado"),
        "spy_return": w.get("spy_ret"),
        "macro_evento": w.get("macro_tomorrow"),
    })

    records = _df_to_records_json_safe(payload)

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


def fetch_vix_trades() -> pd.DataFrame:
    resp = supabase.table("vix_trades").select("*").order("id", desc=True).execute()
    if getattr(resp, "error", None):
        # si no existe, devolvemos vacío sin romper
        return pd.DataFrame()
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    return df


# =============================
# Pipeline
# =============================

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)

    # upserts a TUS tablas
    upsert_vix_daily(out)
    upsert_vix_signal(out)

    return out
