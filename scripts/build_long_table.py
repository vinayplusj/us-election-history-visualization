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


OUT_PATH = Path("data/election_bars_long.csv")

CACHE_DIR = Path("sources_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_text(url: str, cache_path: Path, force_refresh: bool = False) -> str:
    """
    Download a URL and cache it locally (committing cache is optional).
    """
    if cache_path.exists() and not force_refresh:
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
    
# Winners by state sources (National Archives Electoral College pages)
NARA_URLS = {
    2008: "https://www.archives.gov/electoral-college/2008",
    2012: "https://www.archives.gov/electoral-college/2012",
    2016: "https://www.archives.gov/electoral-college/2016",
    2020: "https://www.archives.gov/electoral-college/2020",
    2024: "https://www.archives.gov/electoral-college/2024",
}


def flatten_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip() and str(x).strip().lower() != "nan"]).strip()
            for tup in df.columns
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_int_cell(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    s = s.replace(",", "")
    s = re.sub(r"\[[^\]]*\]", "", s)   # remove [1] style notes
    s = re.sub(r"\([^)]*\)", "", s)    # remove (note) style text
    s = s.strip()
    if s in {"", "-", "—"}:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def normalise_state_name(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z .]", "", str(s)).strip()
    if s2.upper() in {"DC", "D.C.", "DISTRICT OF COLUMBIA"}:
        return "District of Columbia"
    return s2


def extract_party_map_from_nara_html(html: str) -> dict[str, str]:
    """
    Tries to map candidate names to party letters, based on patterns like:
    President John Doe [D]
    Main Opponent Jane Roe [R]
    """
    party_map: dict[str, str] = {}

    for m in re.finditer(r"(President|Main Opponent)\s+([^<\[]+?)\s*\[([A-Z])\]", html):
        name = m.group(2).strip()
        party = m.group(3).strip()
        party_map[name] = party

    return party_map

FALLBACK_YEAR_PARTY_HINTS = {
    2008: {"Democratic": ["Obama"], "Republican": ["McCain"]},
    2012: {"Democratic": ["Obama"], "Republican": ["Romney"]},
    2016: {"Democratic": ["Clinton"], "Republican": ["Trump"]},
    2020: {"Democratic": ["Biden"], "Republican": ["Trump"]},
    2024: {"Democratic": ["Harris", "Democratic"], "Republican": ["Trump", "Republican"]},
}

def party_from_candidate_label(label: str, party_map: dict[str, str], year: int) -> str:
    if not label:
        return "Other"

    label_l = label.lower()

    # First try party_map extracted from page
    for name, party in party_map.items():
        if name.lower() in label_l:
            if party == "D":
                return "Democratic"
            if party == "R":
                return "Republican"
            return "Other"

    # Fallback: year-based hints
    hints = FALLBACK_YEAR_PARTY_HINTS.get(year, {})
    for party_name, tokens in hints.items():
        for token in tokens:
            if token.lower() in label_l:
                return party_name

    return "Other"



def load_state_winners_from_nara() -> pd.DataFrame:
    """
    Returns columns: State, Year, Winning_Party

    Notes:
    - For Maine and Nebraska, the NARA table can show split electoral votes.
      This method assigns the winner as the candidate with the most electoral votes in that state.
    """
    rows: list[dict[str, object]] = []

    for year in YEARS:
        url = NARA_URLS[year]

        # Use requests so we can set User-Agent and also reuse your caching pattern
        html = fetch_text(url, CACHE_DIR / f"nara_{year}.html")

        party_map = extract_party_map_from_nara_html(html)

        # Read tables from HTML
        tables = pd.read_html(StringIO(html))
        if not tables:
            raise ValueError(f"No HTML tables found on NARA page for {year}.")

        # Find the most likely table: has a State column and includes Alabama somewhere
        target = None
        best_score = -1

        for t in tables:
            t = flatten_columns_df(t.copy())
            cols = [str(c).strip().lower() for c in t.columns]

            score = 0
            if "state" in cols:
                score += 10
            # Look for common signal words
            if any("electoral" in c for c in cols):
                score += 2

            # Look for Alabama in the body as a strong signal
            body_text = " ".join(t.astype(str).fillna("").values.flatten().tolist())
            if "Alabama" in body_text:
                score += 5

            # Prefer wider tables (usually candidate columns)
            score += min(len(cols), 20) * 0.1

            if score > best_score:
                best_score = score
                target = t

        if target is None:
            raise ValueError(f"Could not identify the Electoral College by-state table for {year}.")

        # Identify state column
        cols_lower = {str(c).strip().lower(): c for c in target.columns}
        state_col = cols_lower.get("state", target.columns[0])

        df = target.copy()
        df[state_col] = df[state_col].astype(str).str.strip()
        df["State"] = df[state_col].apply(normalise_state_name)

        # Drop totals / empty rows
        df = df[df["State"].str.len() > 0]
        df = df[~df["State"].str.lower().isin(["total"])]

        # Candidate columns: numeric-looking columns except the state column
        candidate_cols: list[str] = []
        for c in df.columns:
            c_l = str(c).strip().lower()
            if c == state_col or c_l == "state":
                continue
        
            # Exclude obvious non-candidate columns
            if any(k in c_l for k in ["total", "electoral", "votes", "vote", "electors", "district", "at large"]):
                continue
        
            sample = df[c].head(20).astype(str)
            hits = sample.str.contains(r"^\s*[\d,]+(\.\d+)?\s*$|^\s*-\s*$|^\s*—\s*$").mean()
            if hits >= 0.5:
                candidate_cols.append(c)


        if not candidate_cols:
            raise ValueError(f"Could not identify candidate columns for {year}. Columns: {list(df.columns)}")

        # Convert candidate columns to ints
        for c in candidate_cols:
            df[c] = df[c].apply(clean_int_cell)

        # Winner label is the candidate column with the maximum EV in that row
        df["Winner_Label"] = df[candidate_cols].idxmax(axis=1)

        # Map candidate label to party
        df["Winning_Party"] = df["Winner_Label"].apply(lambda x: party_from_candidate_label(str(x), party_map, year))    
        # Add year and export
        df["Year"] = year

        out = df[["State", "Year", "Winning_Party"]].copy()

        # NARA pages sometimes include territories in some formats; you can filter if needed.
        out = out.drop_duplicates(subset=["State", "Year"])

        for _, r in out.iterrows():
            rows.append({"State": r["State"], "Year": int(r["Year"]), "Winning_Party": r["Winning_Party"]})

    return pd.DataFrame(rows).drop_duplicates(subset=["State", "Year"])
    


def main() -> None:
    turnout = load_turnout_vep()
    winners = load_state_winners_from_nara()

    final = turnout.merge(winners, on=["State", "Year"], how="left")
    expected_years = set(YEARS)
    counts = final.groupby("Year")["State"].nunique()
    
    bad_years = {y: int(counts.get(y, 0)) for y in expected_years if int(counts.get(y, 0)) != 51}
    if bad_years:
        raise ValueError(f"Expected 51 jurisdictions (50 states + DC). Got: {bad_years}")


    # If a winner is missing, label as Other so Tableau still renders
    final["Winning_Party"] = final["Winning_Party"].fillna("Other")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final = final.sort_values(["State", "Year"])
    final.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(final):,} rows to {OUT_PATH}")



if __name__ == "__main__":
    main()
