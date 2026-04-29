"""
ML state: loads models/data once at startup, exposes transform + similarity search.

Data files are resolved from DATA_DIR env var, defaulting to the project root
(one level above this file), which is where the pkl/csv files currently live.
"""

import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent)))

FEATURE_COLS = [
    "height_inches", "weight", "age_at_draft",
    "combine_max_vertical", "combine_lane_agility", "combine_shuttle",
    "combine_three_qtr_sprint", "combine_wingspan_inches", "wingspan_over_height",
    "pg_g", "pg_mp", "pg_fg_pct", "pg_ft_pct",
    "p36_pts", "p36_reb", "p36_ast", "p36_blk", "p36_stl", "p36_to", "p36_pf",
    "adv1_ts_pct", "adv1_3pa_rate", "adv1_fta_rate",
    "adv1_proj_nba_3p", "adv1_usg_pct", "adv1_ast_usg", "adv1_ast_to",
    "adv2_per", "adv2_ows_40", "adv2_dws_40", "adv2_obpm", "adv2_dbpm",
]


class _State:
    df: pd.DataFrame | None = None
    pca_cols: list[str] | None = None
    pca_matrix: np.ndarray | None = None
    scaler = None
    pca = None
    reducer = None
    medians: dict | None = None


state = _State()


def load_all() -> None:
    state.df = pd.read_csv(DATA_DIR / "players_umap_3d.csv")
    with open(DATA_DIR / "scaler.pkl", "rb") as f:
        state.scaler = pickle.load(f)
    with open(DATA_DIR / "pca.pkl", "rb") as f:
        state.pca = pickle.load(f)
    with open(DATA_DIR / "umap_model.pkl", "rb") as f:
        state.reducer = pickle.load(f)

    # Detect PCA columns dynamically in case n_components differs from 15
    state.pca_cols = sorted(
        [c for c in state.df.columns if re.match(r"^pca_\d+$", c)],
        key=lambda x: int(x.split("_")[1]),
    )
    state.pca_matrix = state.df[state.pca_cols].values

    # Training medians for imputing missing prospect inputs
    state.medians = {col: float(state.df[col].median()) for col in FEATURE_COLS}


def clean_name(raw: str) -> str:
    """'Firstname Lastname POS | School' → 'Firstname Lastname'"""
    return raw.split("|")[0].strip().rsplit(" ", 1)[0]


def height_display(inches: float | None) -> str:
    if inches is None or (isinstance(inches, float) and np.isnan(inches)):
        return "N/A"
    ft = int(inches) // 12
    inch = inches - ft * 12
    return f"{ft}'{inch:.0f}\""


def _safe(v):
    """NaN/inf → None for JSON serialization."""
    if v is None:
        return None
    try:
        if np.isnan(float(v)) or np.isinf(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


def player_to_dict(row: pd.Series) -> dict:
    return {
        "name": clean_name(str(row["name"])),
        "school": row["name"].split("|")[1].strip() if "|" in str(row["name"]) else "",
        "position": str(row["position"]),
        "draft_year": int(row["draft_year"]),
        "pick": int(row["pick"]),
        "limited_data": bool(int(row.get("limited_data") or 0)),
        "umap_x": float(row["umap_x"]),
        "umap_y": float(row["umap_y"]),
        "umap_z": float(row["umap_z"]),
        "stats": {
            "p36_pts": _safe(row.get("p36_pts")),
            "p36_reb": _safe(row.get("p36_reb")),
            "p36_ast": _safe(row.get("p36_ast")),
            "p36_blk": _safe(row.get("p36_blk")),
            "p36_stl": _safe(row.get("p36_stl")),
            "adv1_ts_pct": _safe(row.get("adv1_ts_pct")),
            "adv1_usg_pct": _safe(row.get("adv1_usg_pct")),
            "adv2_obpm": _safe(row.get("adv2_obpm")),
            "adv2_dbpm": _safe(row.get("adv2_dbpm")),
            "height_inches": _safe(row.get("height_inches")),
            "weight": _safe(row.get("weight")),
            "age_at_draft": _safe(row.get("age_at_draft")),
            "adv1_proj_nba_3p": _safe(row.get("adv1_proj_nba_3p")),
        },
    }


def name_to_slug(raw: str) -> str:
    clean = raw.split("|")[0].strip()
    name_only = clean.rsplit(" ", 1)[0]
    return name_only.lower().replace(" ", "-").replace("'", "")


def find_similar(query_pca: np.ndarray, n: int = 3, skip_first: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Similarity in PCA space (not UMAP — UMAP distorts distances)."""
    dists = np.linalg.norm(state.pca_matrix - query_pca, axis=1)
    top_idx = np.argsort(dists)
    top_idx = top_idx[1:n + 1] if skip_first else top_idx[:n]
    return top_idx, dists[top_idx]


def transform_prospect(features: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    features: dict keyed by FEATURE_COLS; missing/None values are imputed
    with training medians before scaling.

    Returns (pca_vec shape (15,), umap_coords shape (3,)).
    """
    vec = np.array([[
        features[col] if features.get(col) is not None else state.medians[col]
        for col in FEATURE_COLS
    ]], dtype=float)

    # Catch explicit NaN values
    for i, col in enumerate(FEATURE_COLS):
        if np.isnan(vec[0, i]):
            vec[0, i] = state.medians[col]

    scaled = state.scaler.transform(vec)
    pca_vec = state.pca.transform(scaled)        # (1, n_components)
    umap_coords = state.reducer.transform(pca_vec)  # (1, 3)
    return pca_vec[0], umap_coords[0]
