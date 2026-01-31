from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


YEARS = [2008, 2012, 2016, 2020, 2024]

# Turnout sources (VEP turnout by state)
TURNOUT_URL_1980_2022 = "https://election.lab.ufl.edu/data-downloads/turnoutdata/Turnout_1980_2022_v1.2.csv"
TURNOUT_URL_2024 = "https://election.lab.ufl.edu/data-downloads/turnoutdata/Turnout_2024G_v0.3.csv"

# Winners by state sources (Wikipedia results tables)
WIKI_URLS = {
    2008: "https://en.wikipedia.org/wiki/2008_United_States_presidential_election",
    2012: "https://en.wikipedia.org/wiki/2012_United_States_presidential_election",
    2016: "https://en.wikipedia.org/wiki/2016_United_States_presidential_election",
    2020: "https://en.wikipedia.org/wiki/2020_United_States_presidential_election",
    2024: "https://en.wikipedia.org/wiki/2024_United_States_presidential_election",
}

OUT_PATH = Path("data/election_bars_long.csv")

CACHE_DIR = Path("sources_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_text(url: str, cache_path: Path) -> str:
    """
    Download a URL and cache it locally (committing cache is optional).
    """
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    headers = {"User-Agent": "us-election-history-tableau (GitHub Actions)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    text = resp.text
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_percent(value) -> float | None:
    """
    Convert values like '61.46%' or 61.46 into a float.
    Returns None if it cannot parse.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    s = s.replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        return float(s)
    except ValueError:
        return None


def load_turnout_vep() -> pd.DataFrame:
    """
    Returns columns: State, Year, Voter_Percentage
    Where Voter_Percentage is VEP turnout percent (0-100).
    """
    # Download both turnout files
    t1 = fetch_text(TURNOUT_URL_1980_2022, CACHE_DIR / "Turnout_1980_2022_v1.2.csv")
    t2 = fetch_text(TURNOUT_URL_2024, CACHE_DIR / "Turnout_2024G_v0.3.csv")

    df1 = pd.read_csv(StringIO(t1))
    df2 = pd.read_csv(StringIO(t2))

    df = pd.concat([df1, df2], ignore_index=True)

    # Try to standardize common column names used in UF turnout tables.
    # These files include a VEP turnout rate column, typically as a percent string.
    col_year = None
    for c in df.columns:
        if str(c).strip().lower() == "year":
            col_year = c
            break
    if col_year is None:
        raise ValueError("Could not find a Year column in turnout files.")

    # State name column can vary. Often it is "State" or similar.
    col_state = None
    for c in df.columns:
        if str(c).strip().lower() == "state":
            col_state = c
            break
    if col_state is None:
        raise ValueError("Could not find a State column in turnout files.")

    # VEP turnout rate column varies. Look for something containing "vep" and "turnout".
    vep_col_candidates = []
    for c in df.columns:
        name = str(c).strip().lower()
        if "vep" in name and "turnout" in name:
            vep_col_candidates.append(c)

    if not vep_col_candidates:
        # Some versions place the VEP turnout rate as the last columns, often two rates (VEP, VAP).
        # As a fallback, search for columns that look like percentages and contain 'vep' nearby.
        raise ValueError(
            "Could not find a VEP turnout column (expected a column containing both 'VEP' and 'turnout')."
        )

    # Choose the first matching VEP turnout column.
    col_vep = vep_col_candidates[0]

    out = df[[col_state, col_year, col_vep]].copy()
    out.columns = ["State", "Year", "Voter_Percentage"]
    
    # Coerce non-numeric years to NaN, then drop them
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["Year"]).copy()
    out["Year"] = out["Year"].astype(int)


    out["Voter_Percentage"] = out["Voter_Percentage"].apply(parse_percent)

    out = out[out["Year"].isin(YEARS)].dropna(subset=["Voter_Percentage"])

    # Normalise state naming where needed
    out["State"] = out["State"].astype(str).str.strip()
    out.loc[out["State"].str.upper().isin(["DC", "D.C.", "DISTRICT OF COLUMBIA"]), "State"] = "District of Columbia"

    # Keep 50 states + DC only
    # UF file should already have the correct set, but we keep this as a safety filter.
    out = out.drop_duplicates(subset=["State", "Year"])

    return out


def flatten_columns(cols) -> list[str]:
    if isinstance(cols, pd.MultiIndex):
        flat = []
        for tup in cols:
            parts = [str(x).strip() for x in tup if str(x).strip() and str(x).strip().lower() != "nan"]
            flat.append(" ".join(parts).strip())
        return flat
    return [str(c).strip() for c in cols]


def infer_party_from_col(colname: str) -> str:
    s = colname.lower()
    # Common parties that can appear in Wikipedia state tables
    if "democratic" in s:
        return "Democratic"
    if "republican" in s:
        return "Republican"
    if "libertarian" in s:
        return "Libertarian"
    if "green" in s:
        return "Green"
    if "constitution" in s:
        return "Constitution"
    if "independent" in s:
        return "Independent"
    return "Other"


def load_state_winners_from_wikipedia() -> pd.DataFrame:
    """
    Returns columns: State, Year, Winning_Party
    """
    rows = []

    for year in YEARS:
        url = WIKI_URLS[year]
        html = fetch_text(url, CACHE_DIR / f"wiki_{year}.html")

        # read_html needs lxml/html5lib/bs4 in the environment (we install those in Actions).
        tables = pd.read_html(StringIO(html))

        # Choose the table that looks like "Results by state"
        target = None
        for t in tables:
            cols = [str(c).lower() for c in flatten_columns(t.columns)]
            if any("state" == c.strip() for c in cols) and any("electoral" in c for c in cols):
                target = t
                break

        if target is None:
            raise ValueError(f"Could not find a 'Results by state' table on Wikipedia for {year}.")

        target.columns = flatten_columns(target.columns)

        # Identify State column
        state_col = None
        for c in target.columns:
            if str(c).strip().lower() == "state":
                state_col = c
                break
        if state_col is None:
            # Sometimes it is "State Total" or similar, but typically "State" exists.
            raise ValueError(f"Could not locate State column in {year} table.")

        df = target.copy()
        df[state_col] = df[state_col].astype(str).str.strip()

        # Keep 50 states + DC, and remove header/footnote rows
        df = df[~df[state_col].str.contains("total", case=False, na=False)]
        df = df[df[state_col].str.len() > 1]

        # Find percent columns. Wikipedia tables include many "%", one per candidate.
        percent_cols = [c for c in df.columns if "%" in str(c)]
        if not percent_cols:
            raise ValueError(f"No percent columns detected in Wikipedia state table for {year}.")

        # Build a party score per row using the percent columns
        party_scores = {}
        for c in percent_cols:
            party = infer_party_from_col(str(c))
            party_scores.setdefault(party, []).append(c)

        def winning_party(row) -> str:
            best_party = "Other"
            best_val = -1.0

            for party, cols in party_scores.items():
                # A party may have multiple % columns in the table (rare), take max.
                vals = []
                for pc in cols:
                    v = parse_percent(row.get(pc))
                    if v is not None:
                        vals.append(v)
                if not vals:
                    continue
                m = max(vals)
                if m > best_val:
                    best_val = m
                    best_party = party

            return best_party

        df["Winning_Party"] = df.apply(winning_party, axis=1)

        # Normalise DC naming
        df.loc[df[state_col].str.upper().isin(["DC", "D.C.", "DISTRICT OF COLUMBIA"]), state_col] = "District of Columbia"

        for _, r in df.iterrows():
            rows.append(
                {"State": r[state_col], "Year": year, "Winning_Party": r["Winning_Party"]}
            )

    out = pd.DataFrame(rows).drop_duplicates(subset=["State", "Year"])
    return out


def main() -> None:
    turnout = load_turnout_vep()
    winners = load_state_winners_from_wikipedia()

    final = turnout.merge(winners, on=["State", "Year"], how="left")

    # Safety: if a winner is missing, label as Other (still colours in Tableau)
    final["Winning_Party"] = final["Winning_Party"].fillna("Other")

    # Output long format for Tableau
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final = final.sort_values(["State", "Year"])
    final.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(final):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
