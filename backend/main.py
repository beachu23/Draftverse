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

import inspect
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


def _build_prompt(prospect: ProspectInput, pca_vec) -> str:
    def _fmt(v, fmt=".1f", suffix=""):
        return f"{v:{fmt}}{suffix}" if v is not None else "N/A"

    def _pct(v):
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    pc1, pc2, pc3, pc4, pc5, pc6 = (float(pca_vec[i]) for i in range(6))
    pc9 = float(pca_vec[8])

    has_combine = any(getattr(prospect, f) is not None for f in [
        "combine_max_vertical", "combine_lane_agility",
        "combine_shuttle", "combine_three_qtr_sprint",
    ])

    combine_note = "" if has_combine else \
        "\n(No combine data for this prospect — PC5 is inferred from college stats only; treat with caution)"

    combine_stats = ""
    if has_combine:
        combine_stats = (
            f"\nCombine: Vertical={_fmt(prospect.combine_max_vertical, '.1f', '\"')} | "
            f"Lane Agility={_fmt(prospect.combine_lane_agility, '.2f', 's')} | "
            f"3/4 Sprint={_fmt(prospect.combine_three_qtr_sprint, '.2f', 's')} | "
            f"Shuttle={_fmt(prospect.combine_shuttle, '.2f', 's')}"
        )

    data_notes = []
    if not has_combine:
        data_notes.append("No combine data available — athletic scores (PC5) are inferred from college production only.")
    if prospect.pg_g is not None and prospect.pg_g < 20:
        data_notes.append(f"Small sample size ({int(prospect.pg_g)} games played) — treat all signals with caution.")
    data_quality_note = " ".join(data_notes) if data_notes else "Full data set available. All signals are based on complete college stats."

    return f"""You are an elite NBA draft analyst writing a concise scouting archetype profile. Your job is to translate statistical patterns into vivid, accurate basketball characterizations. You must be specific and grounded — no clichés, no hype.

=== PCA TRAIT SCORES ===
These scores come from a machine learning model trained on NBA draft prospects 2010-2025. Each component captures a distinct basketball dimension. Scores are standardized (mean=0, std=1). A score of +2.0 is elite. +1.0 is above average. 0.0 is league average. -1.0 is below average. -2.0 is notably weak.

PC1 ({pc1:+.2f}) — SIZE & INTERIOR DOMINANCE  [26.5% of total variance — most important]
High positive: Big, heavy, dominates rebounds and blocks, interior-oriented
High negative: Smaller, perimeter-oriented, high 3-point attempt rate

PC2 ({pc2:+.2f}) — OFFENSIVE STAR POWER  [14.6% of variance]
High positive: Elite offensive producer — high PER, OBPM, win shares, scoring efficiency
Low/negative: Role player production, limited offensive creation

PC3 ({pc3:+.2f}) — TWO-WAY PLAYMAKING  [8.2% of variance]
High positive: Playmaker and defender — high assists, steals, DBPM
High negative: Off-ball specialist, limited creation or defensive activity

PC4 ({pc4:+.2f}) — USAGE & VOLUME vs EFFICIENCY  [6.8% of variance]
High positive: High-usage volume scorer, creates own shot, higher turnovers
High negative: Low-usage, efficient role player, clean decision-making

PC5 ({pc5:+.2f}) — ATHLETIC EXPLOSIVENESS  [5.8% of variance]
High positive: Explosive vertical leap, elite jumping ability
High negative: Slower sprint and agility scores (lower times = more explosive){combine_note}

PC6 ({pc6:+.2f}) — LENGTH & MATURITY  [4.3% of variance]
High positive: Exceptional wingspan relative to height, older/more seasoned prospect
High negative: Shorter wingspan ratio, younger/rawer prospect

PC9 ({pc9:+.2f}) — DURABILITY & AVAILABILITY  [3.1% of variance]
High positive: Played a full season consistently, durable
High negative: Limited games played — injury, eligibility, or small sample

PC7, PC8, PC10-PC15: Not reliable enough to interpret — ignore these entirely.

=== RAW STAT ANCHORS ===
Use these to confirm or contradict the PCA signals above. Only reference a trait if BOTH the PCA score AND the raw stats support it.

Height: {ml.height_display(prospect.height_inches)}  |  Weight: {_fmt(prospect.weight, '.0f', ' lbs')}  |  Age at Draft: {_fmt(prospect.age_at_draft)}
Points/36: {_fmt(prospect.p36_pts)}  |  Rebounds/36: {_fmt(prospect.p36_reb)}  |  Assists/36: {_fmt(prospect.p36_ast)}
Blocks/36: {_fmt(prospect.p36_blk)}  |  Steals/36: {_fmt(prospect.p36_stl)}  |  Turnovers/36: {_fmt(prospect.p36_to)}
True Shooting%: {_pct(prospect.adv1_ts_pct)}  |  Usage%: {_pct(prospect.adv1_usg_pct)}
OBPM: {_fmt(prospect.adv2_obpm)}  |  DBPM: {_fmt(prospect.adv2_dbpm)}
Projected NBA 3P%: {_pct(prospect.adv1_proj_nba_3p)}{combine_stats}

=== DATA QUALITY NOTE ===
{data_quality_note}

=== INSTRUCTIONS ===

STEP 1 — CHOOSE AN ANIMAL
Pick one animal that captures this player's complete basketball archetype. The animal must feel inevitable, not forced.

Rules for the animal:
- Must be a real animal (no mythological creatures)
- Should capture the dominant PC1/PC2 traits primarily
- A panther is all upside. A hyena is scrappy and effective but ugly. A wolf is a pack hunter. A gazelle is fast but fragile. Be honest.
- Avoid the most clichéd picks (lion, eagle, shark) unless truly perfect
- For high-PC1 (interior/big) players, differentiate within the tier — a hippo is an immovable physical presence, a bear is powerful but mobile, a moose is tall and awkward-athletic, a rhino charges through contact, an ox is a tireless workhorse, a bison is a bruising force. Don't default to the same animal for every big.

STEP 2 — WRITE THE ARCHETYPE LABEL
3-5 words capturing their basketball identity.
Examples: "Explosive Two-Way Playmaker", "Efficient Interior Anchor", "Perimeter Creator With Range", "Raw Athletic Upside Project"
Do not default to "High-Volume" as a descriptor unless usage is genuinely the most defining trait — describe basketball identity, not workload.

STEP 3 — WRITE THE SCOUTING PROFILE
Exactly 4 sentences:

Sentence 1: "This player is a [animal] — [one vivid phrase]." The phrase must capture their basketball essence in specific terms.
Sentence 2: Primary strength. Ground this in the highest absolute PC score confirmed by raw stats. Be specific — name the actual skill.
Sentence 3: Secondary trait or how they create value. Reference the animal naturally here.
Sentence 4: If the data shows a clear weakness (a PC score below -0.6 confirmed by raw stats), name it honestly. If no meaningful weakness is evident, end with a forward-looking statement about what this animal needs to prove at the next level — specific, not generic.

RULES:
- Under 90 words total for the writeup
- Never mention PCA scores directly — translate into basketball language
- Never use these words: motor, IQ, upside, intangibles, tools, high-ceiling, work ethic, coachable
- Reference the animal 2-3 times total across all sentences naturally
- Only mention traits supported by BOTH PCA scores AND raw stats
- Be specific and analytical, not hype or generic

=== OUTPUT FORMAT ===
Output exactly this structure, nothing else before or after:
ANIMAL: [animal name in ALL CAPS]
ARCHETYPE: [3-5 word label in Title Case]
WRITEUP: [your 3 sentences, under 110 words]"""


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
    prompt = _build_prompt(prospect, pca_vec)

    async def generate():
        try:
            gemini = genai.Client(api_key=api_key)
            async for chunk in await gemini.aio.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"[GEMINI_ERROR: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")
