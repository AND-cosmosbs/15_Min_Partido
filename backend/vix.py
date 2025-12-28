# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

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

    # “umbrales por percentil”
    q_calm: float = 0.25
    q_tension: float = 0.65
    q_panic: float = 0.85

    # guardarraíl anti “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_vix_floor: float = 12.5  # si VIX < 12.5 => no abrir SVIX


DEFAULT_CFG = VixConfig()


# =============================
# Helpers
# =============================

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_date_str(x) -> Optional[str]:
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return None
        return d.date().isoformat()
    except Exception:
        return None


def _as_bool(x) -> Optional[bool]:
    if x is None or pd.isna(x):
        return None
    return bool(x)


def _as_float(x) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _resilient_upsert(table: str, records: List[Dict[str, Any]], on_conflict: str) -> None:
    """
    Upsert robusto:
    - Convierte date/datetime a string ISO.
    - Si PostgREST devuelve PGRST204 (columna no existe),
      elimina esa clave de todos los records y reintenta.
    """
    if not records:
        return

    # normaliza fechas / tipos no serializables
    cleaned: List[Dict[str, Any]] = []
    for r in records:
        rr = {}
        for k, v in r.items():
            if isinstance(v, (pd.Timestamp,)):
                rr[k] = v.date().isoformat()
            elif hasattr(v, "isoformat") and k == "fecha":
                # datetime.date
                rr[k] = v.isoformat()
            else:
                rr[k] = v
        cleaned.append(rr)

    # reintentos quitando columnas desconocidas
    for _ in range(12):
        resp = supabase.table(table).upsert(cleaned, on_conflict=on_conflict).execute()
        err = getattr(resp, "error", None)
        if not err:
            return

        # error tipo: Could not find the 'contango_estado' column ...
        code = err.get("code") if isinstance(err, dict) else None
        msg = err.get("message") if isinstance(err, dict) else str(err)

        if code == "PGRST204" and "Could not find the" in msg and " column" in msg:
            # extrae el nombre entre comillas simples
            missing = None
            try:
                missing = msg.split("Could not find the '", 1)[1].split("'", 1)[0]
            except Exception:
                missing = None

            if missing:
                for rr in cleaned:
                    rr.pop(missing, None)
                continue

        raise RuntimeError(f"Error upsert en {table}: {err}")

    raise RuntimeError(f"Error upsert en {table}: demasiados reintentos por columnas inexistentes.")


# =============================
# Macro events (opcional)
# =============================

def _fetch_macro_events_safe() -> pd.DataFrame:
    """
    Si no existe la tabla macro_events, devolvemos vacío y listo.
    Estructura esperada: macro_events(fecha, label, impacto, activo)
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
        return df
    except Exception:
        return pd.DataFrame()


def _macro_tomorrow_flag(date_ts: pd.Timestamp, macro_df: pd.DataFrame) -> bool:
    if macro_df is None or macro_df.empty:
        return False
    tomorrow = (pd.to_datetime(date_ts) + pd.Timedelta(days=1)).date()
    w = macro_df.copy()
    if "activo" in w.columns:
        w = w[w["activo"] == True]
    if "fecha" not in w.columns:
        return False
    return (w["fecha"] == tomorrow).any()


# =============================
# Yahoo download
# =============================

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de:
      ^VIX, ^VXN, VIXY, SPY
    Devuelve df con columnas: date, vix, vxn, vixy, spy
    """
    import yfinance as yf  # lazy import

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
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo ({start}..{end})")

        s = data["Close"].copy()
        s.name = col
        df = s.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        out = df if out is None else out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


# =============================
# Features + estado
# =============================

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()
    w["vix"] = _safe_num(w.get("vix", pd.Series(dtype="float")))
    w["vxn"] = _safe_num(w.get("vxn", pd.Series(dtype="float")))
    w["vixy"] = _safe_num(w.get("vixy", pd.Series(dtype="float")))
    w["spy"] = _safe_num(w.get("spy", pd.Series(dtype="float")))

    w["spy_return"] = w["spy"].pct_change()
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    lb = int(cfg.lookback_pct)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(0.25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(0.50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(0.65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(0.85)

    w["vixy_ma_3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma_10"] = w["vixy"].rolling(10).mean()

    # contango estado (texto)
    w["contango_estado"] = pd.NA
    ok_ma = w["vixy_ma_3"].notna() & w["vixy_ma_10"].notna()
    w.loc[ok_ma & (w["vixy_ma_3"] < w["vixy_ma_10"]), "contango_estado"] = "CONTANGO"
    w.loc[ok_ma & (w["vixy_ma_3"] == w["vixy_ma_10"]), "contango_estado"] = "TRANSICION"
    w.loc[ok_ma & (w["vixy_ma_3"] > w["vixy_ma_10"]), "contango_estado"] = "BACKWARDATION"

    # vix regime (texto) según percentiles
    w["vix_regime"] = pd.NA
    ok_pct = w["vix"].notna() & w["vix_p25"].notna() & w["vix_p65"].notna() & w["vix_p85"].notna()
    w.loc[ok_pct & (w["vix"] < w["vix_p25"]), "vix_regime"] = "CALMA"
    w.loc[ok_pct & (w["vix"] >= w["vix_p25"]) & (w["vix"] < w["vix_p65"]), "vix_regime"] = "ALERTA"
    w.loc[ok_pct & (w["vix"] >= w["vix_p65"]) & (w["vix"] < w["vix_p85"]), "vix_regime"] = "TENSION"
    w.loc[ok_pct & (w["vix"] >= w["vix_p85"]), "vix_regime"] = "PANICO"

    return w


def decide_state_row(row: pd.Series, macro_tomorrow: bool, cfg: VixConfig = DEFAULT_CFG) -> Tuple[str, str]:
    """
    Devuelve (estado, motivo) para vix_signal.
    estado: 'SVIX' | 'NEUTRAL' | 'UVIX' | 'CERRAR_UVIX'
    """
    vix = row.get("vix")
    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    ratio = row.get("vxn_vix_ratio")
    ratio_up = bool(row.get("ratio_up")) if pd.notna(row.get("ratio_up")) else False

    contango = row.get("contango_estado")
    spy_ret = row.get("spy_return")

    # sin percentiles todavía => neutral
    if pd.isna(vix) or pd.isna(p25) or pd.isna(p65) or pd.isna(p85):
        return "NEUTRAL", "Insuficiente histórico para percentiles (rolling 252)."

    # guardarraíl VIX demasiado bajo
    if cfg.use_guardrail and pd.notna(vix) and vix < cfg.guardrail_vix_floor:
        return "NEUTRAL", f"Guardarraíl: VIX < {cfg.guardrail_vix_floor} (riesgo snapback)."

    # SVIX (todas)
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and (contango == "CONTANGO")
        and (macro_tomorrow is False)
    )
    if cond_svix:
        return "SVIX", "VIX < P25 + ratio VXN/VIX < 1.25 + contango + sin macro mañana."

    # UVIX (mínimo 2)
    uv1 = vix > p65
    uv2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uv3 = contango == "BACKWARDATION"
    uv4 = (pd.notna(spy_ret) and float(spy_ret) < -0.008)

    score = sum([bool(uv1), bool(uv2), bool(uv3), bool(uv4)])
    if score >= 2:
        return "UVIX", f"Stress (score={score}): vix>p65={uv1}, vxn/vix={uv2}, backward={uv3}, spy<-0.8%={uv4}."

    # CERRAR_UVIX / preparar SVIX
    # (pánico agotándose + vuelve contango + ratio deja de subir)
    if (vix > p85) and (ratio_up is False) and (contango == "CONTANGO"):
        return "CERRAR_UVIX", "VIX > P85 y ratio deja de subir y vuelve contango."

    return "NEUTRAL", "Régimen mixto / transición."


def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df_feat.copy()

    macro = _fetch_macro_events_safe()
    w["macro_evento"] = w["date"].apply(
        lambda d: _macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False
    )

    estados: List[str] = []
    motivos: List[str] = []
    for _, r in w.iterrows():
        stt, mot = decide_state_row(r, macro_tomorrow=bool(r.get("macro_evento")), cfg=cfg)
        estados.append(stt)
        motivos.append(mot)

    w["estado"] = estados
    w["motivo"] = motivos
    return w


# =============================
# Supabase I/O (tus tablas)
# =============================

def upsert_vix_daily(df: pd.DataFrame) -> int:
    """
    Tabla: vix_daily (la tuya)
    PRIMARY KEY: fecha
    Columnas objetivo (si existen):
      fecha, vix, vxn_vix_ratio,
      vix_p25, vix_p50, vix_p65, vix_p85,
      vix_regime,
      vixy_ma_3, vixy_ma_10,
      contango_estado
    """
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = w["date"].apply(_to_date_str)

    records: List[Dict[str, Any]] = []
    for _, r in w.iterrows():
        rec = {
            "fecha": r.get("fecha"),
            "vix": _as_float(r.get("vix")),
            "vxn_vix_ratio": _as_float(r.get("vxn_vix_ratio")),
            "vix_p25": _as_float(r.get("vix_p25")),
            "vix_p50": _as_float(r.get("vix_p50")),
            "vix_p65": _as_float(r.get("vix_p65")),
            "vix_p85": _as_float(r.get("vix_p85")),
            "vix_regime": r.get("vix_regime"),
            "vixy_ma_3": _as_float(r.get("vixy_ma_3")),
            "vixy_ma_10": _as_float(r.get("vixy_ma_10")),
            "contango_estado": r.get("contango_estado"),
        }
        # quita None de fecha
        if rec["fecha"] is None:
            continue
        records.append(rec)

    _resilient_upsert("vix_daily", records, on_conflict="fecha")
    return len(records)


def upsert_vix_signal(df: pd.DataFrame) -> int:
    """
    Tabla: vix_signal (la tuya)
    PRIMARY KEY: fecha
    Columnas: fecha, estado, motivo, vix, vxn_vix_ratio, contango_estado, spy_return, macro_evento
    """
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = w["date"].apply(_to_date_str)

    records: List[Dict[str, Any]] = []
    for _, r in w.iterrows():
        rec = {
            "fecha": r.get("fecha"),
            "estado": r.get("estado") or "NEUTRAL",
            "motivo": r.get("motivo"),
            "vix": _as_float(r.get("vix")),
            "vxn_vix_ratio": _as_float(r.get("vxn_vix_ratio")),
            "contango_estado": r.get("contango_estado"),
            "spy_return": _as_float(r.get("spy_return")),
            "macro_evento": _as_bool(r.get("macro_evento")) if r.get("macro_evento") is not None else False,
        }
        if rec["fecha"] is None:
            continue
        records.append(rec)

    _resilient_upsert("vix_signal", records, on_conflict="fecha")
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


# =============================
# Pipeline
# =============================

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)

    # grabamos a tus tablas
    upsert_vix_daily(out)
    upsert_vix_signal(out)

    return out
