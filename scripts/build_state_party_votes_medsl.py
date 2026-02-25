from __future__ import annotations

import json
import os
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import zipfile
import pandas as pd
import requests

# ----------------------------
# Configuration
# ----------------------------

PERSISTENT_ID = "doi:10.7910/DVN/VOQCHQ"  # MEDSL county presidential returns 2000–2024
DATAVERSE_DATASET_API = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId="
DATAVERSE_FILE_API = "https://dataverse.harvard.edu/api/access/datafile/"

TARGET_YEARS = [2004, 2008, 2012, 2016, 2020, 2024]

JURISDICTIONS_51 = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming", "District of Columbia",
}

OUT_PATH = Path("data/state_party_votes_long_medsl_2004_2024.csv")
CACHE_DIR = Path("sources_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "us-election-history-visualization (GitHub Actions)"}

# Nominee fallback mapping (used only when party fields are missing or messy)
# These are last-name tokens.
NOMINEES: Dict[int, Dict[str, Iterable[str]]] = {
    2004: {"Democratic": ["Kerry"], "Republican": ["Bush"]},
    2008: {"Democratic": ["Obama"], "Republican": ["McCain"]},
    2012: {"Democratic": ["Obama"], "Republican": ["Romney"]},
    2016: {"Democratic": ["Clinton"], "Republican": ["Trump"]},
    2020: {"Democratic": ["Biden"], "Republican": ["Trump"]},
    2024: {"Democratic": ["Harris"], "Republican": ["Trump"]},
}


# ----------------------------
# Helpers
# ----------------------------

def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def fetch_json(url: str, cache_path: Path, force_refresh: bool) -> dict:
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_bytes(url: str, cache_path: Path, force_refresh: bool) -> bytes:
    if cache_path.exists() and not force_refresh:
        return cache_path.read_bytes()

    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    b = resp.content
    cache_path.write_bytes(b)
    return b


def to_int(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace(",", "")
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def to_state_name(s: str) -> str:
    s2 = str(s or "").strip()
    if s2.upper() in {"DC", "D.C.", "DISTRICT OF COLUMBIA"}:
        return "District of Columbia"
    return s2


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in df.columns}
    for want in candidates:
        if want.lower() in cols_l:
            return cols_l[want.lower()]
    return None


def classify_party_from_party_fields(row: pd.Series, party_cols: Iterable[str]) -> Optional[str]:
    for c in party_cols:
        v = row.get(c, None)
        if pd.isna(v):
            continue
        t = normalize_text(v)

        # Common labels in election datasets
        if "democrat" in t:
            return "Democratic"
        if "republican" in t:
            return "Republican"

        # Some datasets use short codes
        if t in {"dem", "democratic"}:
            return "Democratic"
        if t in {"rep", "gop", "republican"}:
            return "Republican"

    return None


def classify_party_from_candidate(row: pd.Series, year: int, candidate_col: str) -> Optional[str]:
    cand = normalize_text(row.get(candidate_col, ""))
    if not cand:
        return None

    hints = NOMINEES.get(year, {})
    for party, tokens in hints.items():
        for tok in tokens:
            if tok.lower() in cand:
                return party
    return None


def find_medsl_main_datafile_id(dataset_json: dict) -> Tuple[int, str]:
    """
    Find the most likely "main" data file in the Dataverse dataset.

    Dataverse often provides tab-delimited (.tab) instead of .csv.
    We pick the largest file among preferred extensions: .tab, .tsv, .csv, .zip.
    Returns (datafile_id, filename).
    """
    data = dataset_json.get("data", {})
    latest = data.get("latestVersion", {})
    files = latest.get("files", [])

    preferred_ext = (".tab", ".tsv", ".csv", ".zip")

    best_id: Optional[int] = None
    best_name: str = ""
    best_size: int = -1

    for f in files:
        df = f.get("dataFile", {})
        file_id = df.get("id")
        filename = (df.get("filename", "") or "").strip()
        filesize = df.get("filesize", -1)

        if file_id is None or not filename:
            continue

        if not filename.lower().endswith(preferred_ext):
            continue

        if isinstance(filesize, int) and filesize > best_size:
            best_id = int(file_id)
            best_name = filename
            best_size = int(filesize)

    if best_id is None:
        # Fallback: pick the largest file in the dataset even if extension is unknown.
        for f in files:
            df = f.get("dataFile", {})
            file_id = df.get("id")
            filename = (df.get("filename", "") or "").strip()
            filesize = df.get("filesize", -1)
            if file_id is None or not filename:
                continue
            if isinstance(filesize, int) and filesize > best_size:
                best_id = int(file_id)
                best_name = filename
                best_size = int(filesize)

    if best_id is None:
        raise ValueError("Could not find a suitable datafile in the MEDSL Dataverse dataset.")

    return best_id, best_name


def read_medsl_table(content: bytes, filename: str) -> pd.DataFrame:
    """
    Read the downloaded datafile bytes into a DataFrame.
    Supports .csv, .tsv, .tab, and .zip containing a single csv/tsv/tab.
    """
    name_l = filename.lower()

    def read_text_table(text: str, sep: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(text), sep=sep)

    # If zipped, extract the first csv/tsv/tab-like file and read it.
    if name_l.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(content)) as z:
            members = [m for m in z.namelist() if m.lower().endswith((".csv", ".tsv", ".tab"))]
            if not members:
                raise ValueError("ZIP did not contain a .csv, .tsv, or .tab file.")
            inner_name = members[0]
            inner_bytes = z.read(inner_name)
            return read_medsl_table(inner_bytes, inner_name)

    # Dataverse .tab is tab-delimited text
    if name_l.endswith(".tab") or name_l.endswith(".tsv"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        return read_text_table(text, sep="\t")

    # Default to CSV
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    return read_text_table(text, sep=",")


# ----------------------------
# Main build
# ----------------------------

def build_state_party_votes(force_refresh: bool) -> pd.DataFrame:
    # 1) Fetch dataset metadata (cached)
    meta_url = DATAVERSE_DATASET_API + requests.utils.quote(PERSISTENT_ID, safe="")
    meta_cache = CACHE_DIR / "medsl_dv_dataset_meta.json"
    meta = fetch_json(meta_url, meta_cache, force_refresh=force_refresh)

    # 2) Find the primary CSV datafile
    file_id, filename = find_medsl_main_datafile_id(meta)
    print(f"MEDSL Dataverse file chosen: id={file_id}, filename={filename}")

    # 3) Download the datafile (cached)
    file_url = DATAVERSE_FILE_API + str(file_id)
    file_cache = CACHE_DIR / f"medsl_{file_id}_{Path(filename).name}"
    content = fetch_bytes(file_url, file_cache, force_refresh=force_refresh)

    df = read_medsl_table(content, filename)
    if df.empty:
        raise ValueError("MEDSL datafile loaded but it is empty.")

    # 4) Identify columns we need (robust, case-insensitive)
    year_col = pick_first_existing(df, ["year", "election_year"])
    state_col = pick_first_existing(df, ["state", "state_name", "state_po", "state_postal"])
    votes_col = pick_first_existing(df, ["candidatevotes", "votes", "totalvotes", "candidate_votes", "vote"])
    candidate_col = pick_first_existing(df, ["candidate", "candidate_name", "cand", "nominee"])

    # Party fields differ by version; try common names
    party_detailed_col = pick_first_existing(df, ["party_detailed", "party_detail", "party"])
    party_simplified_col = pick_first_existing(df, ["party_simplified", "party_simple", "party"])

    missing = [("year", year_col), ("state", state_col), ("votes", votes_col)]
    missing = [k for k, v in missing if v is None]
    if missing:
        raise ValueError(f"MEDSL file missing required columns: {missing}. Columns found: {list(df.columns)}")

    # 5) Normalize and filter to target years
    work = df.copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=[year_col]).copy()
    work[year_col] = work[year_col].astype(int)
    work = work[work[year_col].isin(TARGET_YEARS)].copy()

    # 6) Build party classification
    party_cols = []
    if party_detailed_col is not None:
        party_cols.append(party_detailed_col)
    if party_simplified_col is not None and party_simplified_col not in party_cols:
        party_cols.append(party_simplified_col)

    def party_label(row: pd.Series) -> Optional[str]:
        # First, use party fields if present
        p = classify_party_from_party_fields(row, party_cols)
        if p is not None:
            return p

        # Next, use candidate matching if candidate column exists
        if candidate_col is not None:
            y = int(row[year_col])
            p2 = classify_party_from_candidate(row, y, candidate_col=candidate_col)
            if p2 is not None:
                return p2

        return None

    work["Party2"] = work.apply(party_label, axis=1)
    work = work[work["Party2"].isin(["Democratic", "Republican"])].copy()

    # 7) Clean and aggregate votes
    work["Votes"] = pd.to_numeric(work[votes_col], errors="coerce")
    work = work.dropna(subset=["Votes"]).copy()
    work["Votes"] = work["Votes"].astype(int)

    # Normalize state name
    # MEDSL may have state postal codes or full names. Prefer full names if present.
    # If we get postal codes, we need a lookup. We will detect and convert if needed.
    work["StateRaw"] = work[state_col].astype(str).str.strip()

    # Detect if StateRaw looks like postal codes (two letters) for most rows
    sample = work["StateRaw"].head(200)
    postal_ratio = (sample.str.fullmatch(r"[A-Z]{2}") | sample.str.fullmatch(r"[a-z]{2}")).mean()

    if postal_ratio >= 0.80:
        # Postal code to state name mapping
        abv_to_state = {
            "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware",
            "FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana",
            "ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana",
            "NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina",
            "ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina",
            "SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia",
            "WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia",
        }
        work["State"] = work["StateRaw"].str.upper().map(abv_to_state).fillna(work["StateRaw"])
    else:
        work["State"] = work["StateRaw"].apply(to_state_name)

    # Aggregate
    agg = (
        work.groupby(["State", year_col, "Party2"], as_index=False)["Votes"]
        .sum()
        .rename(columns={year_col: "Year"})
    )

    # Pivot to Dem_Votes and Rep_Votes
    pivot = agg.pivot_table(index=["State", "Year"], columns="Party2", values="Votes", aggfunc="sum").reset_index()
    pivot.columns = [c if isinstance(c, str) else str(c) for c in pivot.columns]

    if "Democratic" not in pivot.columns or "Republican" not in pivot.columns:
        raise ValueError(f"Pivot did not produce both Democratic and Republican columns. Columns: {list(pivot.columns)}")

    pivot = pivot.rename(columns={"Democratic": "Dem_Votes", "Republican": "Rep_Votes"})
    pivot["Dem_Votes"] = pivot["Dem_Votes"].fillna(0).astype(int)
    pivot["Rep_Votes"] = pivot["Rep_Votes"].fillna(0).astype(int)

    # Filter to canonical 51
    pivot["State"] = pivot["State"].astype(str).str.strip()
    pivot.loc[pivot["State"].str.upper().isin(["DC", "D.C.", "DISTRICT OF COLUMBIA"]), "State"] = "District of Columbia"
    pivot = pivot[pivot["State"].isin(JURISDICTIONS_51)].copy()

    # Ensure complete grid (51 x 6)
    expected_rows = 51 * len(TARGET_YEARS)
    got_states = set(pivot["State"].tolist())
    missing_states = sorted(set(JURISDICTIONS_51) - got_states)
    if missing_states:
        raise ValueError(f"Missing states after aggregation: {missing_states}")

    # Check year completeness per state
    counts = pivot.groupby("Year")["State"].nunique().to_dict()
    bad = {y: counts.get(y, 0) for y in TARGET_YEARS if counts.get(y, 0) != 51}
    if bad:
        raise ValueError(f"Expected 51 states per year. Got: {bad}")

    if len(pivot) != expected_rows:
        # This can happen if there are duplicate state-year rows. Enforce uniqueness.
        pivot = pivot.groupby(["State", "Year"], as_index=False)[["Dem_Votes", "Rep_Votes"]].sum()
        if len(pivot) != expected_rows:
            raise ValueError(f"Expected exactly {expected_rows} rows, got {len(pivot)}.")

    # Check non-zero totals
    zeros = pivot[(pivot["Dem_Votes"] <= 0) | (pivot["Rep_Votes"] <= 0)]
    if not zeros.empty:
        sample = zeros.head(10)[["State", "Year", "Dem_Votes", "Rep_Votes"]]
        raise ValueError(
            "Found non-positive Dem or Rep votes in some rows. Sample:\n"
            + sample.to_string(index=False)
        )

    pivot = pivot.sort_values(["State", "Year"]).reset_index(drop=True)
    return pivot


def main() -> None:
    force_refresh = env_flag("FORCE_REFRESH", default=False)

    out = build_state_party_votes(force_refresh=force_refresh)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(out):,} rows to {OUT_PATH}")
    print("Counts by year:")
    print(out.groupby("Year")["State"].nunique().reindex(TARGET_YEARS).to_string())


if __name__ == "__main__":
    main()
