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
        return {
            "estado": "NEUTRAL",
            "accion": "NO DATA",
            "comentario": "Insuficiente histórico para rolling 252."
        }

    # ---------------------------------------------------------
    # Guardarraíl VIX demasiado bajo
    # ---------------------------------------------------------
    if cfg.use_guardrail and float(vix) < float(cfg.guardrail_vix_floor):
        return {
            "estado": "NEUTRAL",
            "accion": "NO OPEN SVIX",
            "comentario": "Guardarraíl: VIX extremadamente bajo."
        }

    # ---------------------------------------------------------
    # SVIX (NO SE TOCA)
    # ---------------------------------------------------------
    svix_thr = p10 if pd.notna(p10) else p25
    spy_filter_ok = True
    if pd.notna(spy_ret):
        spy_filter_ok = float(spy_ret) > -0.007

    cond_svix = (
        (vix < svix_thr)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (not macro_tomorrow)
        and spy_filter_ok
    )
    if cond_svix:
        return {
            "estado": "SVIX",
            "accion": "OPEN/HOLD SVIX",
            "comentario": "Calma extrema + contango + SPY ok + sin macro."
        }

    # ---------------------------------------------------------
    # UVIX — PÁNICO REAL (ENDURECIDO)
    # ---------------------------------------------------------
    uvix_cond_1 = (pd.notna(vix) and pd.notna(p85) and vix > p85)
    uvix_cond_2 = (pd.notna(spy_ret) and spy_ret <= -0.015)
    uvix_cond_3 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uvix_cond_4 = (
        pd.notna(row.get("vixy_ma_3"))
        and pd.notna(row.get("vixy_ma_10"))
        and row.get("vixy_ma_3") > row.get("vixy_ma_10")
    )

    if uvix_cond_1 and uvix_cond_2 and uvix_cond_3 and uvix_cond_4:
        return {
            "estado": "UVIX",
            "accion": "OPEN/HOLD UVIX",
            "comentario": "PÁNICO REAL: VIX>p85 + SPY<-1.5% + ratio alto + estructura rota."
        }

    # ---------------------------------------------------------
    # PREP_SVIX
    # ---------------------------------------------------------
    cond_prep = (vix > p85) and (not ratio_up) and contango_ok
    if cond_prep:
        return {
            "estado": "PREP_SVIX",
            "accion": "WAIT / PREPARE SVIX",
            "comentario": "Pánico se enfría + contango vuelve."
        }

    return {
        "estado": "NEUTRAL",
        "accion": "NO NEW POSITION",
        "comentario": "Régimen mixto / transición."
    }
