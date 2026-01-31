from __future__ import annotations

import io
import os
from pathlib import Path
import pandas as pd
import requests

OUT_PATH = Path("data/election_bars_long.csv")

YEARS = [2008, 2012, 2016, 2020, 2024]

def download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def standardize_state_names(df: pd.DataFrame, col: str = "State") -> pd.DataFrame:
    df[col] = df[col].astype(str).str.strip()
    return df

def load_turnout_vep() -> pd.DataFrame:
    """
    Return columns:
      State, Year, Voter_Percentage
    Voter_Percentage should be numeric (example: 61.8, not "61.8%").
    """
    frames = []

    # TODO: Replace these URLs with the turnout-by-state VEP% sources you choose.
    # Recommended: one file that already includes all years, or one per year.
    #
    # Example patterns:
    #   url = f"https://example.com/turnout_state_{year}.csv"
    #
    for year in YEARS:
        url = os.environ.get(f"TURNOUT_URL_{year}")
        if not url:
            raise ValueError(f"Missing environment variable TURNOUT_URL_{year}")
        df = download_csv(url)

        # TODO: Rename columns to match your source
        # Example:
        # df = df.rename(columns={"state": "State", "vep_turnout_pct": "Voter_Percentage"})
        # df["Year"] = year

        df = df[["State", "Year", "Voter_Percentage"]].copy()
        frames.append(df)

    turnout = pd.concat(frames, ignore_index=True)
    turnout = standardize_state_names(turnout, "State")

    turnout["Year"] = turnout["Year"].astype(int)
    turnout["Voter_Percentage"] = pd.to_numeric(turnout["Voter_Percentage"], errors="coerce")

    return turnout

def load_winners_2008_2020() -> pd.DataFrame:
    """
    Return columns:
      State, Year, Winning_Party
    Compute by grouping official vote totals by State+Year+Party, then choose max votes.
    """
    # TODO: Replace with your chosen results dataset URL (state-level or county-level).
    results_url = os.environ.get("RESULTS_URL_2008_2020")
    if not results_url:
        raise ValueError("Missing environment variable RESULTS_URL_2008_2020")

    df = download_csv(results_url)

    # TODO: Rename to match your source columns.
    # Required fields:
    #   State, Year, Party, Votes
    # Example:
    # df = df.rename(columns={"state": "State", "year": "Year", "party": "Party", "candidatevotes": "Votes"})

    df = standardize_state_names(df, "State")
    df["Year"] = df["Year"].astype(int)
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)

    df = df[df["Year"].isin([2008, 2012, 2016, 2020])].copy()

    grouped = df.groupby(["State", "Year", "Party"], as_index=False)["Votes"].sum()
    winners = grouped.sort_values(["State", "Year", "Votes"], ascending=[True, True, False])
    winners = winners.groupby(["State", "Year"], as_index=False).head(1)

    winners = winners.rename(columns={"Party": "Winning_Party"})[["State", "Year", "Winning_Party"]]
    return winners

def load_winners_2024() -> pd.DataFrame:
    """
    Return columns:
      State, Year, Winning_Party

    Easiest robust approach:
    - Maintain scripts/winners_2024.csv as a curated mapping of State -> Winning_Party.
    - Keep it small and stable.
    """
    path = Path("scripts/winners_2024.csv")
    if not path.exists():
        raise FileNotFoundError("Missing scripts/winners_2024.csv")

    df = pd.read_csv(path)
    df = standardize_state_names(df, "State")
    df["Year"] = 2024
    df = df[["State", "Year", "Winning_Party"]]
    return df

def main() -> None:
    turnout = load_turnout_vep()
    winners_2008_2020 = load_winners_2008_2020()
    winners_2024 = load_winners_2024()

    winners = pd.concat([winners_2008_2020, winners_2024], ignore_index=True)

    long_df = turnout.merge(winners, on=["State", "Year"], how="inner")
    long_df = long_df.sort_values(["State", "Year"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()
