"""
NBA Draft Prospect Similarity API
===================================
Three endpoints:
  GET  /players              — all 462 players (umap coords + stats) + training medians
  POST /similarity           — prospect stats → 3 nearest comps + prospect umap coords
  POST /similarity/blurb     — same input → streaming Gemini analysis (SSE)

Startup: loads CSV + pkl files once into memory via ml.load_all().
Env vars: GEMINI_API_KEY (required for /blurb), CORS_ORIGINS (default "*"), DATA_DIR.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from google import genai
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import ml

# Load .env from project root (parent of backend/)
load_dotenv(Path(__file__).parent.parent / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml.load_all()
    yield


app = FastAPI(title="NBA Draft Prospect Similarity API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ──────────────────────────────────────────────────────────────

class ProspectInput(BaseModel):
    # Display / prompt only — not a feature column
    position: Optional[str] = None

    # Feature columns — all optional; missing values imputed server-side
    height_inches: Optional[float] = None
    weight: Optional[float] = None
    age_at_draft: Optional[float] = None
    combine_max_vertical: Optional[float] = None
    combine_lane_agility: Optional[float] = None
    combine_shuttle: Optional[float] = None
    combine_three_qtr_sprint: Optional[float] = None
    combine_wingspan_inches: Optional[float] = None
    pg_g: Optional[float] = None
    pg_mp: Optional[float] = None
    pg_fg_pct: Optional[float] = None
    pg_ft_pct: Optional[float] = None
    p36_pts: Optional[float] = None
    p36_reb: Optional[float] = None
    p36_ast: Optional[float] = None
    p36_blk: Optional[float] = None
    p36_stl: Optional[float] = None
    p36_to: Optional[float] = None
    p36_pf: Optional[float] = None
    adv1_ts_pct: Optional[float] = None
    adv1_3pa_rate: Optional[float] = None
    adv1_fta_rate: Optional[float] = None
    adv1_proj_nba_3p: Optional[float] = None
    adv1_usg_pct: Optional[float] = None
    adv1_ast_usg: Optional[float] = None
    adv1_ast_to: Optional[float] = None
    adv2_per: Optional[float] = None
    adv2_ows_40: Optional[float] = None
    adv2_dws_40: Optional[float] = None
    adv2_obpm: Optional[float] = None
    adv2_dbpm: Optional[float] = None

    n_results: int = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prepare_features(prospect: ProspectInput) -> dict:
    features = prospect.model_dump(exclude={"position", "n_results"})
    wi = features.get("combine_wingspan_inches")
    hi = features.get("height_inches")
    features["wingspan_over_height"] = (wi - hi) if (wi is not None and hi is not None) else None
    return features


def _build_prompt(prospect: ProspectInput, comps: list[dict]) -> str:
    def _fmt(v, fmt=".1f", suffix=""):
        return f"{v:{fmt}}{suffix}" if v is not None else "N/A"

    def _pct(v):
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    def _comp_stats(s: dict) -> str:
        parts = []
        if s.get("p36_pts") is not None:
            parts.append(f"{s['p36_pts']:.1f} pts")
        if s.get("p36_reb") is not None:
            parts.append(f"{s['p36_reb']:.1f} reb")
        if s.get("p36_ast") is not None:
            parts.append(f"{s['p36_ast']:.1f} ast")
        if s.get("adv1_ts_pct") is not None:
            parts.append(f"{s['adv1_ts_pct'] * 100:.1f}% TS")
        return ", ".join(parts) if parts else "limited stats"

    comp_lines = "\n".join(
        f"{i + 1}. {c['name']} ({c['draft_year']}, #{c['pick']}) — {_comp_stats(c['stats'])}"
        for i, c in enumerate(comps)
    )

    return f"""A prospect has entered the NBA draft with the following profile:
- Position: {prospect.position or 'Unknown'}
- Height: {ml.height_display(prospect.height_inches)}, Weight: {_fmt(prospect.weight, '.0f', ' lbs')}, Age: {_fmt(prospect.age_at_draft)}
- Points/36: {_fmt(prospect.p36_pts)}, Rebounds/36: {_fmt(prospect.p36_reb)}, Assists/36: {_fmt(prospect.p36_ast)}
- TS%: {_pct(prospect.adv1_ts_pct)}, USG%: {_pct(prospect.adv1_usg_pct)}, OBPM: {_fmt(prospect.adv2_obpm)}, DBPM: {_fmt(prospect.adv2_dbpm)}
- Projected NBA 3P%: {_pct(prospect.adv1_proj_nba_3p)}

Their 3 closest historical draft comps are:
{comp_lines}

In 2-3 sentences, explain what these comps reveal about this prospect's archetype, strengths, and what kind of NBA player they project to be. Be specific and analytical, not generic. Do not use filler phrases."""


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/players")
def get_players():
    """Return all 462 players with umap coords, stats, and training medians."""
    players = [ml.player_to_dict(row) for _, row in ml.state.df.iterrows()]
    return {"players": players, "medians": ml.state.medians}


@app.get("/players/{slug}/similar")
def get_player_similar(slug: str):
    """Return 3 nearest neighbors for an existing player identified by slug."""
    slugs = ml.state.df["name"].apply(ml.name_to_slug)
    matches = slugs[slugs == slug]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Player '{slug}' not found")
    row_idx = int(matches.index[0])
    pca_vec = ml.state.pca_matrix[row_idx]
    top_idx, _ = ml.find_similar(pca_vec, n=3, skip_first=True)
    return {"similar": [ml.player_to_dict(ml.state.df.iloc[int(i)]) for i in top_idx]}


@app.post("/similarity")
def get_similarity(prospect: ProspectInput):
    """
    Transform prospect stats → PCA → UMAP.
    Return prospect's umap coords + n_results nearest comps (PCA-space distance).
    """
    features = _prepare_features(prospect)
    pca_vec, umap_coords = ml.transform_prospect(features)
    top_idx, dists = ml.find_similar(pca_vec, n=prospect.n_results)

    comps = [
        {**ml.player_to_dict(ml.state.df.iloc[int(idx)]), "distance": float(dist)}
        for idx, dist in zip(top_idx, dists)
    ]

    return {
        "prospect": {
            "umap_x": float(umap_coords[0]),
            "umap_y": float(umap_coords[1]),
            "umap_z": float(umap_coords[2]),
        },
        "comps": comps,
    }


@app.post("/similarity/blurb")
async def stream_blurb(prospect: ProspectInput):
    """
    Compute comps internally, build prompt, stream Gemini analysis word by word.
    Can be called in parallel with /similarity — both do the same ML transform.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured")

    features = _prepare_features(prospect)
    pca_vec, _ = ml.transform_prospect(features)
    top_idx, _ = ml.find_similar(pca_vec, n=3)

    comps = [ml.player_to_dict(ml.state.df.iloc[int(idx)]) for idx in top_idx]
    prompt = _build_prompt(prospect, comps)

    async def generate():
        try:
            client = genai.Client(api_key=api_key)
            async for chunk in await client.aio.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"[ERROR: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")
