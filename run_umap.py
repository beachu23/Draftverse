"""
UMAP Dimensionality Reduction
==============================
Takes players_processed.csv (output of prepare_data.py) and runs UMAP
on the PCA components to produce 2D coordinates for visualization.

Also runs a quick similarity search test to verify the pipeline works
end-to-end before hooking up to the FastAPI backend.

Usage:
    pip install umap-learn pandas numpy
    python run_umap.py

Output:
    players_umap.csv   — all player data + umap_x, umap_y coordinates
    umap_model.pkl     — fitted UMAP model for transforming new prospects
"""

import pickle
import numpy as np
import pandas as pd
import umap

INPUT_CSV  = "players_processed.csv"
OUTPUT_CSV = "players_umap_3d.csv"
UMAP_PKL   = "umap_model.pkl"

# ── UMAP hyperparameters ───────────────────────────────────────────────────────
# n_neighbors: balances local vs global structure
#   lower  = tighter local clusters, less global context
#   higher = more global structure preserved, looser clusters
#   15 is the standard default and works well for ~460 points
#
# min_dist: how tightly points are packed in 2D
#   lower  = tighter clumps, easier to see clusters
#   higher = more even spread, better for seeing relationships between clusters
#   0.1 gives good cluster definition without overcrowding
#
# metric: euclidean on PCA output is correct since PCA already
#   normalized and decorrelated the features
#
# random_state: fixed for reproducibility — same data always
#   produces same plot

N_NEIGHBORS  = 15
MIN_DIST     = 0.1
N_COMPONENTS = 3
METRIC       = "euclidean"
RANDOM_STATE = 42


def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} players, {len(df.columns)} columns")

    # ── Extract PCA matrix ─────────────────────────────────────────────────────
    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    X_pca = df[pca_cols].values
    print(f"  Using {len(pca_cols)} PCA components as UMAP input")

    # ── Fit UMAP ───────────────────────────────────────────────────────────────
    print(f"\nFitting UMAP (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST})...")
    reducer = umap.UMAP(
        n_components=N_COMPONENTS,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=METRIC,
        random_state=RANDOM_STATE,
    )
    X_umap = reducer.fit_transform(X_pca)
    print(f"  Done. Output shape: {X_umap.shape}")

    # Save fitted model for transforming new prospects later
    with open(UMAP_PKL, "wb") as f:
        pickle.dump(reducer, f)
    print(f"  Saved UMAP model -> {UMAP_PKL}")

    # ── Add coordinates to dataframe ───────────────────────────────────────────
    df["umap_x"] = X_umap[:, 0]
    df["umap_y"] = X_umap[:, 1]
    df["umap_z"] = X_umap[:, 2]
    df["umap_z"] = X_umap[:, 2]

    # ── Similarity search function ─────────────────────────────────────────────
    # NOTE: similarity is computed in PCA space (not UMAP space)
    # UMAP distorts distances — it's for visualization only.
    # PCA space preserves the true Euclidean distances between players.

    def find_similar(name, n=5):
        row = df[df["name"].str.contains(name, case=False, na=False)]
        if len(row) == 0:
            print(f"  '{name}' not found")
            return
        query_vec = row[pca_cols].values[0]
        dists = np.linalg.norm(df[pca_cols].values - query_vec, axis=1)
        df["_dist"] = dists
        results = df.nsmallest(n + 1, "_dist").iloc[1:]  # skip self
        print(f"\n  {n} most similar to {row.iloc[0]['name'][:40]}:")
        for _, r in results.iterrows():
            school = r["name"].split("|")[-1].strip() if "|" in r["name"] else ""
            print(f"    [{r['draft_year']} #{int(r['pick'])}] "
                  f"{r['name'].split('|')[0].strip():<28} "
                  f"({school:<25}) dist={r['_dist']:.3f}")

    # ── Spot check similarity results ─────────────────────────────────────────
    print("\n=== SIMILARITY SPOT CHECKS ===")
    test_players = [
        "Cooper Flagg",
        "Zion Williamson",
        "John Wall",
        "Karl-Anthony Towns",
        "Steph",        # Stephon Castle
        "Wembanyama",
    ]
    for p in test_players:
        find_similar(p, n=5)

    df.drop(columns=["_dist"], errors="ignore", inplace=True)

    # ── Save output ────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV} ({len(df)} rows)")

    # ── UMAP coordinate summary ────────────────────────────────────────────────
    print("\n=== UMAP COORDINATE RANGES ===")
    print(f"  umap_x: [{df['umap_x'].min():.2f}, {df['umap_x'].max():.2f}]")
    print(f"  umap_y: [{df['umap_y'].min():.2f}, {df['umap_y'].max():.2f}]")
    print(f"  umap_z: [{df['umap_z'].min():.2f}, {df['umap_z'].max():.2f}]")
    print(f"  umap_z: [{df['umap_z'].min():.2f}, {df['umap_z'].max():.2f}]")

    print("\n=== SAMPLE COORDINATES (notable players) ===")
    notables = ["Cooper Flagg", "Zion", "John Wall", "Karl-Anthony Towns",
                "Luka", "Wembanyama", "Giannis", "Kawhi"]
    for p in notables:
        row = df[df["name"].str.contains(p, case=False, na=False)]
        if len(row):
            r = row.iloc[0]
            print(f"  {r['name'].split('|')[0].strip():<28} "
                  f"umap=({r['umap_x']:+.2f}, {r['umap_y']:+.2f}, {r['umap_z']:+.2f})")


if __name__ == "__main__":
    main()