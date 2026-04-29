#!/usr/bin/env python3
"""
Batch-generate animal archetype scouting profiles for all 462 players.
Uses GPT-4o-mini. Saves incrementally to player_archetypes.json.
Resume-safe: skips players already in the JSON.

Setup:
  pip install openai python-dotenv pandas numpy
  Add OPENAI_API_KEY to .env

Usage:
  python generate_archetypes.py
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

ROOT        = Path(__file__).parent
OUTPUT_FILE = ROOT / "player_archetypes.json"

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(ROOT / "players_umap_3d.csv")
pca_cols = sorted(
    [c for c in df.columns if re.match(r"^pca_\d+$", c)],
    key=lambda x: int(x.split("_")[1]),
)
pca_matrix = df[pca_cols].values

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def name_to_slug(raw: str) -> str:
    clean     = raw.split("|")[0].strip()
    name_only = clean.rsplit(" ", 1)[0]
    return name_only.lower().replace(" ", "-").replace("'", "")


def height_display(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    ft   = int(float(val)) // 12
    inch = float(val) - ft * 12
    return f"{ft}'{inch:.0f}\""


def _fmt(val, fmt=".1f", suffix="") -> str:
    if val is None or (isinstance(val, float) and np.isnan(float(val))):
        return "N/A"
    return f"{float(val):{fmt}}{suffix}"


def _pct(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(float(val))):
        return "N/A"
    return f"{float(val) * 100:.1f}%"


def _get(row: pd.Series, col: str):
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(row: pd.Series, pca_vec: np.ndarray) -> str:
    pc1, pc2, pc3, pc4, pc5, pc6 = (float(pca_vec[i]) for i in range(6))
    pc9 = float(pca_vec[8])

    combine_fields = [
        "combine_max_vertical", "combine_lane_agility",
        "combine_shuttle", "combine_three_qtr_sprint",
    ]
    has_combine = any(_get(row, f) is not None for f in combine_fields)

    combine_note = "" if has_combine else \
        "\n(No combine data for this player — PC5 is inferred from college stats only; treat with caution)"

    combine_stats = ""
    if has_combine:
        combine_stats = (
            f"\nCombine: Vertical={_fmt(_get(row, 'combine_max_vertical'), '.1f', chr(34))} | "
            f"Lane Agility={_fmt(_get(row, 'combine_lane_agility'), '.2f', 's')} | "
            f"3/4 Sprint={_fmt(_get(row, 'combine_three_qtr_sprint'), '.2f', 's')} | "
            f"Shuttle={_fmt(_get(row, 'combine_shuttle'), '.2f', 's')}"
        )

    data_notes = []
    limited = bool(int(row.get("limited_data") or 0))
    pg_g    = _get(row, "pg_g")
    if not has_combine:
        data_notes.append("No combine data — athletic scores (PC5) inferred from college stats only.")
    if pg_g is not None and float(pg_g) < 20:
        data_notes.append(f"Small sample size ({int(float(pg_g))} games played) — treat with caution.")
    if limited:
        data_notes.append("Limited data flag: >30% features missing before imputation.")
    data_quality_note = " ".join(data_notes) if data_notes else \
        "Full data set available. All signals based on complete college + combine data."

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

Height: {height_display(_get(row, 'height_inches'))}  |  Weight: {_fmt(_get(row, 'weight'), '.0f', ' lbs')}  |  Age at Draft: {_fmt(_get(row, 'age_at_draft'))}
Points/36: {_fmt(_get(row, 'p36_pts'))}  |  Rebounds/36: {_fmt(_get(row, 'p36_reb'))}  |  Assists/36: {_fmt(_get(row, 'p36_ast'))}
Blocks/36: {_fmt(_get(row, 'p36_blk'))}  |  Steals/36: {_fmt(_get(row, 'p36_stl'))}  |  Turnovers/36: {_fmt(_get(row, 'p36_to'))}
True Shooting%: {_pct(_get(row, 'adv1_ts_pct'))}  |  Usage%: {_pct(_get(row, 'adv1_usg_pct'))}
OBPM: {_fmt(_get(row, 'adv2_obpm'))}  |  DBPM: {_fmt(_get(row, 'adv2_dbpm'))}
Projected NBA 3P%: {_pct(_get(row, 'adv1_proj_nba_3p'))}{combine_stats}

=== DATA QUALITY NOTE ===
{data_quality_note}

=== INSTRUCTIONS ===

STEP 1 — CHOOSE AN ANIMAL
Pick one animal that captures this player's complete basketball archetype — strengths AND limitations. The animal must feel inevitable, not forced.

Rules:
- Must be a real animal (no mythological creatures)
- Capture the dominant PC1/PC2 traits primarily; hint at limitations too
- A panther is all upside. A hyena is scrappy but ugly. A wolf is a pack hunter. A gazelle is fast but fragile. Be honest.
- Avoid the most clichéd picks (lion, eagle, shark) unless truly perfect
- If no PC score exceeds ±1.0, choose an animal that reflects statistical anonymity honestly

STEP 2 — WRITE THE ARCHETYPE LABEL
3-5 words capturing their basketball identity.
If no PC score exceeds ±1.0, reflect that (e.g. "Statistically Unremarkable Role Candidate")

STEP 3 — WRITE THE SCOUTING PROFILE
Exactly 3-4 sentences:
Sentence 1: "This player is a [animal] — [one vivid phrase]."
Sentence 2: Primary strength, grounded in highest absolute PC score confirmed by raw stats.
Sentence 3: Secondary trait. Reference the animal naturally.
Sentence 4: Core limitation, grounded in most negative PC score confirmed by raw stats.

If no score exceeds ±1.0: 2 sentences only, acknowledge no dominant trait.

RULES:
- Under 90 words total for the writeup
- Never mention PCA scores directly — translate into basketball language
- Never use these words: motor, IQ, upside, intangibles, tools, high-ceiling, work ethic, coachable
- Reference the animal 2-3 times naturally across all sentences
- Only mention traits confirmed by BOTH PCA and raw stats
- Be specific and analytical, not hype or generic

=== OUTPUT FORMAT ===
Output exactly this structure, nothing else before or after:
ANIMAL: [ANIMAL NAME IN ALL CAPS]
ARCHETYPE: [3-5 Word Label In Title Case]
WRITEUP: [3-4 sentences, under 90 words]"""


# ── GPT call with retry ───────────────────────────────────────────────────────

def call_gpt(prompt: str, max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  ⚠ API error (attempt {attempt+1}/{max_retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    return None


def parse_response(text: str) -> dict | None:
    animal    = re.search(r"ANIMAL:\s*(.+?)(\n|$)",     text, re.IGNORECASE)
    archetype = re.search(r"ARCHETYPE:\s*(.+?)(\n|$)",  text, re.IGNORECASE)
    writeup   = re.search(r"WRITEUP:\s*([\s\S]*)",      text, re.IGNORECASE)
    if not (animal and archetype and writeup):
        return None
    return {
        "animal":    animal.group(1).strip(),
        "archetype": archetype.group(1).strip(),
        "writeup":   writeup.group(1).strip(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            archetypes = json.load(f)
        print(f"Resuming — {len(archetypes)}/{len(df)} players already done.\n")
    else:
        archetypes = {}

    total      = len(df)
    new_count  = 0
    fail_count = 0

    for idx, row in df.iterrows():
        slug = name_to_slug(str(row["name"]))

        if slug in archetypes:
            continue

        clean = str(row["name"]).split("|")[0].strip().rsplit(" ", 1)[0]
        print(f"[{idx+1}/{total}] {clean}...", end=" ", flush=True)

        raw    = call_gpt(build_prompt(row, pca_matrix[idx]))
        parsed = parse_response(raw) if raw else None

        if parsed is None:
            snippet = repr(raw[:80]) if raw else "None"
            print(f"FAILED — {snippet}")
            fail_count += 1
            continue

        archetypes[slug] = parsed
        new_count += 1
        print(f"✓  {parsed['animal']} / {parsed['archetype']}")

        if new_count % 10 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(archetypes, f, indent=2)

        time.sleep(0.05)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(archetypes, f, indent=2)

    print(f"\nDone. {new_count} new, {fail_count} failed. Total in file: {len(archetypes)}/{total}")


if __name__ == "__main__":
    main()
