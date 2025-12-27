from backend.vix import run_vix_pipeline

df = run_vix_pipeline(start="2025-01-01", end="2026-01-01")
print(df.tail(5)[["date","vix","vxn_vix_ratio","contango_ok","macro_tomorrow","estado","accion"]])
