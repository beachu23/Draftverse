"""
NBA Draft Prospect Data Preparation Pipeline
=============================================
Cleans tankathon_draft_picks.csv and prepares it for PCA + UMAP.

Steps:
    1.  Parse raw string columns to numeric
    2.  Engineer wingspan_over_height
    3.  Select feature columns (drop redundant/identifier cols)
    4.  Drop players missing >60% of feature columns
    5.  Impute remaining NaNs with column median
    6.  Winsorize at 1st/99th percentile per column
    7.  StandardScale
    8.  PCA to 90% explained variance
    9.  Save players_processed.csv, scaler.pkl, pca.pkl

Usage:
    pip install pandas scikit-learn
    python prepare_data.py
"""

import re
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
import os

if len(sys.argv) < 2:
    print("Usage: python prepare_data.py <input_csv>")
    print("Example: python prepare_data.py tankathon_draft_picks.csv")
    sys.exit(1)

INPUT_CSV  = sys.argv[1]
if not os.path.exists(INPUT_CSV):
    print(f"Error: file not found: {INPUT_CSV}")
    sys.exit(1)

OUTPUT_CSV = "players_processed.csv"
SCALER_PKL = "scaler.pkl"
PCA_PKL    = "pca.pkl"

MISSING_THRESHOLD = 0.60
WINSOR_LOW        = 0.01
WINSOR_HIGH       = 0.99

# Kept as labels alongside features — not fed into the model
ID_COLS = ["draft_year", "pick", "name", "position"]

# Final feature set agreed upon
FEATURE_COLS = [
    # Bio
    "height_inches",
    "weight",
    "age_at_draft",
    # Combine
    "combine_max_vertical",
    "combine_lane_agility",
    "combine_shuttle",
    "combine_three_qtr_sprint",
    "combine_wingspan_inches",
    "wingspan_over_height",
    # Per game (role/context)
    "pg_g",
    "pg_mp",
    "pg_fg_pct",
    "pg_ft_pct",
    # Per 36 (production normalized for minutes)
    "p36_pts",
    "p36_reb",
    "p36_ast",
    "p36_blk",
    "p36_stl",
    "p36_to",
    "p36_pf",
    # Advanced I
    "adv1_ts_pct",
    "adv1_3pa_rate",
    "adv1_fta_rate",
    "adv1_proj_nba_3p",
    "adv1_usg_pct",
    "adv1_ast_usg",
    "adv1_ast_to",
    # Advanced II
    "adv2_per",
    "adv2_ows_40",
    "adv2_dws_40",
    "adv2_obpm",
    "adv2_dbpm",
]


# ── Parsing helpers ────────────────────────────────────────────────────────────

def parse_height(s):
    """
    Convert height/reach/wingspan string to total inches.
    Handles: "6'4\"" "6'10.25\"" "8'5.5\""
    """
    if not isinstance(s, str) or not s.strip() or s.strip() == "nan":
        return np.nan
    s = s.strip().replace("\u2019", "'")
    m = re.match(r"(\d+)'([\d.]+)\"?", s)
    if m:
        return float(m.group(1)) * 12 + float(m.group(2))
    try:
        return float(s)
    except ValueError:
        return np.nan

def parse_weight(s):
    if not isinstance(s, str) or not s.strip() or s.strip() == "nan":
        return np.nan
    try:
        return float(s.replace("lbs", "").strip())
    except ValueError:
        return np.nan

def parse_age(s):
    if not isinstance(s, str) or not s.strip() or s.strip() == "nan":
        return np.nan
    try:
        return float(s.replace("yrs", "").strip())
    except ValueError:
        return np.nan

def parse_combine_measure(s):
    """Strip trailing quote mark from vertical/agility measurements."""
    if not isinstance(s, str) or not s.strip() or s.strip() == "nan":
        return np.nan
    try:
        return float(s.strip().replace('"', ""))
    except ValueError:
        return np.nan

def parse_stat(s):
    if not isinstance(s, str) or not s.strip() or s.strip() == "nan":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ── Winsorization ──────────────────────────────────────────────────────────────

def winsorize(df, cols, low=0.01, high=0.99):
    df = df.copy()
    for col in cols:
        lo = df[col].quantile(low)
        hi = df[col].quantile(high)
        df[col] = df[col].clip(lo, hi)
    return df


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, dtype=str)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # ── Step 1: Parse raw strings ──────────────────────────────────────────────
    print("\nStep 1: Parsing raw columns to numeric...")

    df["height_inches"]           = df["height"].apply(parse_height)
    df["weight"]                  = df["weight"].apply(parse_weight)
    df["age_at_draft"]            = df["age_at_draft"].apply(parse_age)
    df["combine_wingspan_inches"] = df["combine_wingspan"].apply(parse_height)
    df["combine_max_vertical"]    = df["combine_max_vertical"].apply(parse_combine_measure)
    df["combine_lane_agility"]    = df["combine_lane_agility"].apply(parse_stat)
    df["combine_shuttle"]         = df["combine_shuttle"].apply(parse_stat)
    df["combine_three_qtr_sprint"]= df["combine_three_qtr_sprint"].apply(parse_stat)

    for col in ["pg_g","pg_mp","pg_fg_pct","pg_ft_pct",
                "p36_pts","p36_reb","p36_ast","p36_blk","p36_stl","p36_to","p36_pf",
                "adv1_ts_pct","adv1_3pa_rate","adv1_fta_rate","adv1_proj_nba_3p",
                "adv1_usg_pct","adv1_ast_usg","adv1_ast_to",
                "adv2_per","adv2_ows_40","adv2_dws_40","adv2_obpm","adv2_dbpm"]:
        df[col] = df[col].apply(parse_stat)

    # ── Step 2: Engineer wingspan_over_height ──────────────────────────────────
    print("Step 2: Engineering wingspan_over_height...")
    df["wingspan_over_height"] = np.where(
        df["combine_wingspan_inches"].notna() & df["height_inches"].notna(),
        df["combine_wingspan_inches"] - df["height_inches"],
        np.nan
    )

    # ── Step 3: Select columns ─────────────────────────────────────────────────
    print("Step 3: Selecting feature + identifier columns...")
    df = df[ID_COLS + FEATURE_COLS].copy()

    # ── Step 4: Drop players missing >60% of features ──────────────────────────
    print(f"Step 4: Dropping players missing >{MISSING_THRESHOLD*100:.0f}% of features...")
    missing_frac = df[FEATURE_COLS].isna().mean(axis=1)
    mask = missing_frac <= MISSING_THRESHOLD
    dropped = df[~mask]
    df = df[mask].copy()
    print(f"  Kept {len(df)} / {len(df)+len(dropped)} players")
    if len(dropped):
        print(f"  Dropped: {', '.join(dropped['name'].str.split('|').str[0].str.strip().tolist())}")

    # ── Step 4b: Capture limited_data flag BEFORE imputation ─────────────────
    # Players missing >30% of features flagged — mostly international players
    # whose missing advanced stats are all imputed medians (less reliable comps)
    missing_frac_post_drop = df[FEATURE_COLS].isna().mean(axis=1)
    df["limited_data"] = (missing_frac_post_drop > 0.30).astype(int)
    n_limited = df["limited_data"].sum()
    print(f"  {n_limited} players flagged as limited_data (>30% missing before imputation)")

    # ── Step 5: Impute remaining NaNs with column median ──────────────────────
    print("Step 5: Imputing NaNs with column median...")
    medians = df[FEATURE_COLS].median()
    nan_before = df[FEATURE_COLS].isna().sum().sum()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(medians)
    print(f"  Imputed {nan_before} values across {(df[FEATURE_COLS].isna().sum() == 0).sum()} columns")

    # ── Step 6: Winsorize at 1st/99th percentile ──────────────────────────────
    print(f"Step 6: Winsorizing at {WINSOR_LOW*100:.0f}th/{WINSOR_HIGH*100:.0f}th percentile...")
    df = winsorize(df, FEATURE_COLS, WINSOR_LOW, WINSOR_HIGH)
    # Show what got capped
    print(f"  James Wiseman PER after winsorize: {df[df['name'].str.contains('Wiseman', na=False)]['adv2_per'].values}")

    # ── Step 7: StandardScale ──────────────────────────────────────────────────
    print("Step 7: StandardScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved {SCALER_PKL}")

    # ── Step 8: PCA ───────────────────────────────────────────────────────────
    print("Step 8: Running PCA (90% variance threshold)...")
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    n_comp = pca.n_components_
    var_explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  {n_comp} components explain {var_explained*100:.1f}% of variance")
    print(f"  Variance per component: {[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
    with open(PCA_PKL, "wb") as f:
        pickle.dump(pca, f)
    print(f"  Saved {PCA_PKL}")

    # ── Step 9: Save output ────────────────────────────────────────────────────
    print("Step 9: Saving output...")

    # Add PCA components as columns (used for similarity search)
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"pca_{i+1}" for i in range(n_comp)],
        index=df.index
    )

    # Add scaled features (useful for debugging/inspection)
    scaled_df = pd.DataFrame(
        X_scaled,
        columns=[f"scaled_{c}" for c in FEATURE_COLS],
        index=df.index
    )

    out = pd.concat([df[ID_COLS + FEATURE_COLS + ["limited_data"]], scaled_df, pca_df], axis=1)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved {OUTPUT_CSV} ({len(out)} rows, {len(out.columns)} columns)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"  Players in:        480")
    print(f"  Players out:       {len(out)}")
    print(f"  Feature columns:   {len(FEATURE_COLS)}")
    print(f"  PCA components:    {n_comp}")
    print(f"  Variance explained:{var_explained*100:.1f}%")
    print(f"  Output files:      {OUTPUT_CSV}, {SCALER_PKL}, {PCA_PKL}")


if __name__ == "__main__":
    main()