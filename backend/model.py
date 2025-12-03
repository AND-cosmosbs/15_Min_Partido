# backend/model.py

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_historical_data(data_dir: str | Path) -> pd.DataFrame:
    """
    Carga los ficheros all-euro-data-*.xls* desde la carpeta data/
    y devuelve un DataFrame histórico unificado.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("all-euro-data-*.xls*"))

    if not files:
        raise FileNotFoundError(
            f"No se han encontrado ficheros 'all-euro-data-*.xls*' en {data_path.resolve()}"
        )

    dfs: list[pd.DataFrame] = []

    for f in files:
        xl = pd.ExcelFile(f, engine="openpyxl")
        for sh in xl.sheet_names:
            if sh.upper() == "VARIABLES":
                continue
            df = xl.parse(sh)

            # Fecha en formato europeo
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(
                    df["Date"], dayfirst=True, errors="coerce"
                )

            # Rellenar Div si no viene en la hoja
            if "Div" not in df.columns:
                df["Div"] = sh

            dfs.append(df)

    hist = pd.concat(dfs, ignore_index=True)

    # Asegurar numéricos relevantes
    for c in ["HTHG", "HTAG", "B365H", "B365D", "B365A"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

    return hist


def compute_team_and_league_stats(
    hist: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula:
      - team_stats: p(0-0 HT) y T_score por equipo
      - div_stats: p(0-0 HT), goles medias HT y L_score por liga
    """

    # --- Stats por equipo ---
    home = hist.copy()
    home["Team"] = home["HomeTeam"]

    away = hist.copy()
    away["Team"] = away["AwayTeam"]

    long = pd.concat([home, away], ignore_index=True)

    long["is_00_HT"] = (
        (long["HTHG"] == 0) & (long["HTAG"] == 0)
    ).astype(int)

    team_stats = (
        long.groupby("Team")["is_00_HT"]
        .agg(["mean", "count"])
        .reset_index()
    )
    team_stats.columns = ["Team", "p_00_HT", "matches"]

    def team_score(r) -> int:
        p = r["p_00_HT"]
        n = r["matches"]
        pts = 0

        if p >= 0.40:
            pts += 3
        elif p >= 0.33:
            pts += 2
        elif p >= 0.27:
            pts += 1

        if n >= 40:
            pts += 1

        return pts

    team_stats["T_score"] = team_stats.apply(team_score, axis=1)

    # --- Stats por liga ---
    div_stats = (
        hist.groupby("Div")
        .apply(
            lambda g: pd.Series(
                {
                    "avg_g_ht": g["HTHG"].mean() + g["HTAG"].mean(),
                    "p_00_HT": (
                        (g["HTHG"] == 0) & (g["HTAG"] == 0)
                    ).mean(),
                }
            )
        )
        .reset_index()
    )

    def league_score(r) -> int:
        p = r["p_00_HT"]
        g = r["avg_g_ht"]
        pts = 0

        # p(0-0 HT)
        if p >= 0.32:
            pts += 2
        elif p >= 0.28:
            pts += 1

        # goles en 1ª parte
        if g <= 1.05:
            pts += 2
        elif g <= 1.25:
            pts += 1

        return pts

    div_stats["L_score"] = div_stats.apply(league_score, axis=1)

    def league_tier(score: int) -> str:
        if score >= 3:
            return "Oro"
        elif score >= 2:
            return "Plata"
        return "Bronce"

    div_stats["LeagueTier"] = div_stats["L_score"].apply(league_tier)

    return team_stats, div_stats


def score_fixtures(
    team_stats: pd.DataFrame,
    div_stats: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aplica el modelo a un DataFrame de fixtures (tal cual fixtures.xlsx de Football-Data).

    Devuelve un DataFrame con:
      - L_score, LeagueTier
      - H_T_score, A_T_score
      - MatchScore, MatchClass
      - FavOdd, NoClearFav
      - PickType (Ideal / Buena filtrada / None)
    """

    fx = fixtures.copy()

    # Fecha y cuotas a numérico
    if "Date" in fx.columns:
        fx["Date"] = pd.to_datetime(
            fx["Date"], dayfirst=True, errors="coerce"
        )

    for c in ["B365H", "B365D", "B365A"]:
        if c in fx.columns:
            fx[c] = pd.to_numeric(fx[c], errors="coerce")

    # Scores de liga
    fx = fx.merge(
        div_stats[["Div", "L_score", "LeagueTier"]],
        on="Div",
        how="left",
    )

    # Scores de equipo local
    fx = fx.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "HomeTeam", "T_score": "H_T_score"}
        ),
        on="HomeTeam",
        how="left",
    )

    # Scores de equipo visitante
    fx = fx.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "AwayTeam", "T_score": "A_T_score"}
        ),
        on="AwayTeam",
        how="left",
    )

    # Score de partido
    fx["MatchScore"] = (
        fx["L_score"].fillna(0)
        + fx["H_T_score"].fillna(0)
        + fx["A_T_score"].fillna(0)
    )

    def classify(score: float) -> str:
        if score >= 9:
            return "Ideal"
        elif score >= 7:
            return "Buena"
        elif score >= 5:
            return "Borderline"
        return "Descartar"

    fx["MatchClass"] = fx["MatchScore"].apply(classify)

    # Filtro “no favorito claro” (cuota favorito >= 2.0)
    fx["FavOdd"] = fx[["B365H", "B365A"]].min(axis=1)
    fx["NoClearFav"] = fx["FavOdd"] >= 2.0

    # Buena filtrada (cuando no hay Ideal o como segunda capa)
    fx["IsBuenaFiltrada"] = (
        (fx["MatchClass"] == "Buena")
        & (fx["MatchScore"] >= 8)
        & (fx["LeagueTier"].isin(["Oro", "Plata"]))
        & (fx["NoClearFav"])
    )

    # Tipo de pick
    fx["PickType"] = None
    fx.loc[
        (fx["MatchClass"] == "Ideal") & (fx["NoClearFav"]),
        "PickType",
    ] = "Ideal"
    fx.loc[
        fx["IsBuenaFiltrada"] & fx["PickType"].isna(),
        "PickType",
    ] = "Buena filtrada"

    return fx
