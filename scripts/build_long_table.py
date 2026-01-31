from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

import requests

YEARS = [2008, 2012, 2016, 2020, 2024]

JURISDICTIONS_51 = {
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana",
    "Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana",
    "Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina",
    "South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia",
    "Wisconsin","Wyoming","District of Columbia"
}

    
# Winners by state sources (National Archives Electoral College pages)
NARA_URLS = {
    2008: "https://www.archives.gov/electoral-college/2008",
    2012: "https://www.archives.gov/electoral-college/2012",
    2016: "https://www.archives.gov/electoral-college/2016",
    2020: "https://www.archives.gov/electoral-college/2020",
    2024: "https://www.archives.gov/electoral-college/2024",
}

# Turnout sources (VEP turnout by state)
TURNOUT_URL_1980_2022 = "https://election.lab.ufl.edu/data-downloads/turnoutdata/Turnout_1980_2022_v1.2.csv"
TURNOUT_URL_2024 = "https://election.lab.ufl.edu/data-downloads/turnoutdata/Turnout_2024G_v0.3.csv"


OUT_PATH = Path("data/election_bars_long.csv")

CACHE_DIR = Path("sources_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def make_base_grid() -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sorted(JURISDICTIONS_51), YEARS],
        names=["State", "Year"]
    ).to_frame(index=False)

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

    # Standardise df1 (1980–2022)
    # Find year/state/vep turnout columns case-insensitively
    def find_col(df: pd.DataFrame, want: str) -> str:
        for c in df.columns:
            if str(c).strip().lower() == want:
                return c
        raise ValueError(f"Could not find column '{want}' in turnout file.")

    year1 = find_col(df1, "year")
    state1 = find_col(df1, "state")

    vep1_candidates = []
    for c in df1.columns:
        name = str(c).strip().lower()
        if "vep" in name and "turnout" in name:
            vep1_candidates.append(c)
    if not vep1_candidates:
        raise ValueError("Could not find VEP turnout column in 1980–2022 turnout file.")
    vep1 = vep1_candidates[0]

    out1 = df1[[state1, year1, vep1]].copy()
    out1.columns = ["State", "Year", "Voter_Percentage"]
    out1["Year"] = pd.to_numeric(out1["Year"], errors="coerce")
    out1 = out1.dropna(subset=["Year"]).copy()
    out1["Year"] = out1["Year"].astype(int)

    # Standardise df2 (2024) and FORCE Year=2024
    # Many 2024 versions do not include a clean Year column per row.
    state2 = None
    for c in df2.columns:
        if str(c).strip().lower() == "state":
            state2 = c
            break
    if state2 is None:
        raise ValueError("Could not find a State column in 2024 turnout file.")

    vep2_candidates = []
    for c in df2.columns:
        name = str(c).strip().lower()
        if "vep" in name and "turnout" in name:
            vep2_candidates.append(c)
    if not vep2_candidates:
        raise ValueError("Could not find VEP turnout column in 2024 turnout file.")
    vep2 = vep2_candidates[0]

    out2 = df2[[state2, vep2]].copy()
    out2.columns = ["State", "Voter_Percentage"]
    out2["Year"] = 2024

    # Combine
    out = pd.concat([out1, out2], ignore_index=True)

    # Parse percent values
    out["Voter_Percentage"] = out["Voter_Percentage"].apply(parse_percent)

    # If values look like proportions (0-1), convert to percent
    mask = out["Voter_Percentage"].between(0, 1, inclusive="both")
    out.loc[mask, "Voter_Percentage"] = out.loc[mask, "Voter_Percentage"] * 100.0


    # Keep only target years and non-null turnout
    out = out[out["Year"].isin(YEARS)].dropna(subset=["Voter_Percentage"]).copy()

    # Normalise DC naming
    out["State"] = out["State"].astype(str).str.strip()
    out.loc[out["State"].str.upper().isin(["DC", "D.C.", "DISTRICT OF COLUMBIA"]), "State"] = "District of Columbia"

    # Filter to the canonical 51
    out = out[out["State"].isin(JURISDICTIONS_51)].copy()

    # Ensure one row per State-Year
    out = out.drop_duplicates(subset=["State", "Year"], keep="last")

    return out




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
            hits = sample.str.contains(r"^\s*[\d,]+(?:\.\d+)?\s*$|^\s*-\s*$|^\s*—\s*$").mean()
            if hits >= 0.5:
                candidate_cols.append(c)


        if not candidate_cols:
            raise ValueError(f"Could not identify candidate columns for {year}. Columns: {list(df.columns)}")

        # Convert candidate columns to ints
        for c in candidate_cols:
            df[c] = df[c].apply(clean_int_cell)

        # Collapse district rows (Maine / Nebraska) by summing candidate EV within each State
        # This prevents counts like 52 and ensures one record per jurisdiction.
        df = df[df["State"].isin(JURISDICTIONS_51)].copy()

        grouped = df.groupby("State", as_index=False)[candidate_cols].sum()

        grouped["Winner_Label"] = grouped[candidate_cols].idxmax(axis=1)
        grouped["Winning_Party"] = grouped["Winner_Label"].apply(lambda x: party_from_candidate_label(str(x), party_map, year))

        grouped["Year"] = year

        out = grouped[["State", "Year", "Winning_Party"]].copy()



        # NARA pages sometimes include territories in some formats; you can filter if needed.
        out = out.drop_duplicates(subset=["State", "Year"])

        for _, r in out.iterrows():
            rows.append({"State": r["State"], "Year": int(r["Year"]), "Winning_Party": r["Winning_Party"]})

    return pd.DataFrame(rows).drop_duplicates(subset=["State", "Year"])
    


def main() -> None:
    turnout = load_turnout_vep()
    winners = load_state_winners_from_nara()

    base = make_base_grid()

    final = (
        base
        .merge(turnout, on=["State", "Year"], how="left")
        .merge(winners, on=["State", "Year"], how="left")
    )
    
    # If a winner is missing, label as Other so Tableau still renders
    final["Winning_Party"] = final["Winning_Party"].fillna("Other")

    
    expected_years = set(YEARS)
    counts = final.groupby("Year")["State"].nunique()
    
    bad_years = {y: int(counts.get(y, 0)) for y in expected_years if int(counts.get(y, 0)) != 51}
    if bad_years:
        raise ValueError(f"Expected 51 jurisdictions (50 states + DC). Got: {bad_years}")


    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final = final.sort_values(["State", "Year"])
    final.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(final):,} rows to {OUT_PATH}")



if __name__ == "__main__":
    main()
