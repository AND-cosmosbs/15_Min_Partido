import pandas as pd

def classify_fixtures(df, team_stats, league_stats):
    df = df.copy()

    # unir L_score y tier
    df = df.merge(
        league_stats[["Div", "L_score", "LeagueTier"]], 
        on="Div", 
        how="left"
    )

    # unir T_score home
    df = df.merge(
        team_stats.rename(columns={"Team": "HomeTeam", "T_score": "H_T_score"}),
        on="HomeTeam",
        how="left"
    )

    # unir T_score away
    df = df.merge(
        team_stats.rename(columns={"Team": "AwayTeam", "T_score": "A_T_score"}),
        on="AwayTeam",
        how="left"
    )

    df["MatchScore"] = (
        df["L_score"].fillna(0) +
        df["H_T_score"].fillna(0) +
        df["A_T_score"].fillna(0)
    )

    def classify(s):
        if s >= 9: return "Ideal"
        elif s >= 7: return "Buena"
        elif s >= 5: return "Borderline"
        return "Descartar"

    df["MatchClass"] = df["MatchScore"].apply(classify)

    # filtro de favorito
    df["FavOdd"] = df[["B365H", "B365A"]].min(axis=1)
    df["NoClearFav"] = df["FavOdd"] >= 2.0

    # buena filtrada
    df["IsBuenaFiltrada"] = (
        (df["MatchClass"] == "Buena") &
        (df["MatchScore"] >= 8) &
        (df["LeagueTier"].isin(["Oro", "Plata"])) &
        (df["NoClearFav"])
    )

    # asignar pick
    df["PickType"] = None
    df.loc[(df["MatchClass"] == "Ideal") & (df["NoClearFav"]), "PickType"] = "Ideal"
    df.loc[df["IsBuenaFiltrada"] & df["PickType"].isna(), "PickType"] = "Buena filtrada"

    return df
