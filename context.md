# NBA Draft Prospect Similarity App — Project Context

## Concept
A fullstack web app that drops a prospect into a "universe" of NBA draft picks
(2010–2025). Users input a prospect's stats, the app finds their 3 closest
historical comps, and flies the camera through a 3D star field to land on
where the prospect sits in draft history. Each player — inputted or clicked —
gets a Gemini-generated animal archetype + scouting writeup.

---

## User Flow (FINAL)

1. **Landing screen** — clean input form, dark background. `/players` fetch
   runs silently in background. Submit button disabled ("Preparing universe...")
   until fetch completes, then shows "Enter the Draft Universe".

2. **Submit** — form fades out (300ms), universe expands to fill screen.
   Camera begins GSAP snake animation through 3D star field.

3. **Camera lands** — zooms into prospect's position. Prospect appears as
   gold (#ffd700) sprite, slightly larger than other players. 3 nearest comps
   illuminate and enlarge. Comp bubbles appear with name/draft year/key stats.

4. **Prospect bubble** — appears showing:
   - Animal name (large, prominent, top of bubble)
   - Archetype label (smaller font, below animal)
   - Key stats
   - A separate connected bubble appears top-right and streams the
     Gemini scouting writeup word by word (live call)

5. **Exploration** — user freely navigates 3D universe with OrbitControls.
   Clicking any player star:
   - That star pulses and enlarges
   - Their bubble appears with precomputed animal archetype + writeup
   - Their 3 nearest neighbors in PCA space also illuminate and enlarge
     (so you can see who each historical player is most similar to)
   Small "search again" button returns to form.

6. **Search again** — universe fades, input form returns.

---

## Key Design Decisions

- **Visual theme: arcade pixel-art** — the universe looks like a retro arcade
  game (Space Invaders / Galaga aesthetic), not a realistic galaxy. Hard pixel
  borders, pure black void background, phosphor-CRT bloom, no soft gradients
  or nebula. This is the direction going forward — do not revert to a realistic
  space aesthetic.
- **Field star sprites** — pixel-art 4-point stars drawn pixel-by-pixel on a
  32×32 canvas (`imageSmoothingEnabled = false`). Five blue/cyan palette
  variants with hard 1px dark outlines. Each star has a slow staggered ambient
  blink (1.8–5s period, square-wave opacity).
- **Prospect sprite** — pixel-art 8-point star (cardinal + diagonal arms),
  gold (#ffdd00) fill, dark amber outline. Slowly rotates (`material.rotation`)
  and scale-snaps between 26 and 32 every 0.5s after camera lands (discrete
  arcade power-up pulse). Gold particle ring orbits it.
- **Comp sprites** — same 4-point shape as field stars but orange (#ff7700)
  fill with white outline. Scale 18. Blink in sync at 0.45s period, full
  on/off — alert cadence vs the slow ambient field blink.
- **No position color coding** — keeps the arcade aesthetic clean.
- **Navigation: WASD + mouse-drag** — not OrbitControls. After camera fly-in
  completes, user navigates freely with WASD + scroll wheel + click-drag to
  look around.
- **Loading = the form** — Three.js initializes after /players fetch completes,
  not on page load. No spinner needed.
- **Similarity in PCA space, NEVER UMAP space** — UMAP distorts distances.
- **Animal archetype is central to the app identity** — displayed prominently
  on every player bubble. Not a secondary feature.

---

## Tech Stack (FINAL)

### Frontend
- **React** + **Vite**
- **Three.js** — 3D universe, vanilla only (NOT @react-three/fiber)
- **GSAP** — camera snake animation and tweening
- **Tailwind CSS v3** — styling (PostCSS setup)
- **@tanstack/react-query** — data fetching

### Backend
- **FastAPI** — single main.py + ml.py, no database
- **pandas** — CSV loaded into memory at startup (462 rows, static)
- **scikit-learn + umap-learn + numpy** — ML transform on new inputs
- **google-genai** — Gemini 2.5 Flash (NOT google-generativeai)
- **Streaming** — FastAPI StreamingResponse, async generator
- No database, no ORM, no migrations

### Gemini Model
- `gemini-2.5-flash` — confirmed working, 5 RPM / 20 RPD free tier
  (RPM is the binding limit during development — space calls out)
- SDK: `google-genai` (current), NOT deprecated `google-generativeai`
- Streaming pattern:
```python
async def generate():
    client = genai.Client(api_key=api_key)
    async for chunk in await client.aio.models.generate_content_stream(
        model="gemini-2.5-flash", contents=prompt):
        if chunk.text:
            yield chunk.text
return StreamingResponse(generate(), media_type="text/plain")
```

### Hosting
- **Railway** — FastAPI backend (env var: GEMINI_API_KEY)
- **Vercel** — React frontend
- CORS: allow_credentials=False (no auth), allow_origins wildcard ok

---

## Data Pipeline (COMPLETE — do not redo)

### Files on disk
| File | Description |
|---|---|
| `tankathon_draft_picks.csv` | Raw scraped data, 480 rows, 62 cols |
| `players_processed.csv` | Cleaned + scaled + PCA + limited_data, 462 rows, 84 cols |
| `players_umap_3d.csv` | Above + umap_x, umap_y, umap_z, 462 rows |
| `player_archetypes.json` | Precomputed animal+archetype+writeup per player slug |
| `scaler.pkl` | Fitted StandardScaler |
| `pca.pkl` | Fitted PCA (15 components, 90% variance) |
| `umap_model.pkl` | Fitted UMAP (3D, n_neighbors=15, min_dist=0.1, seed=42) |
| `scouting_prompt.py` | Gemini prompt builder using PCA loadings |

### Why 462 not 480
18 players dropped — international/G-League with >60% features missing.
Natural gap: zero players exist between 50-70% missing.
49 players flagged limited_data=1 (>30% missing before imputation) —
these are international players whose advanced stats are imputed medians.

### Pipeline steps
1. Parse raw strings (height→inches, weight strip "lbs", age strip "yrs")
2. Engineer `wingspan_over_height` = combine_wingspan_inches - height_inches
   NOTE: This is subtraction (raw delta in inches), NOT division. Scouts
   use the raw delta — absolute reach advantage matters, not ratio.
3. Drop redundant columns
4. Drop players missing >60% of 32 feature columns
5. Flag limited_data = 1 if >30% missing BEFORE imputation
6. Impute remaining NaNs with column median (not mean — skewed distributions)
7. Winsorize at 1st/99th percentile (handles Wiseman 3-game PER=52.7 → 36.6)
8. StandardScaler → save scaler.pkl
9. PCA → 15 components, 90% variance → save pca.pkl
10. UMAP → 3 components → save umap_model.pkl → players_umap_3d.csv

---

## Feature Columns (32 total)

```python
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
```

Notable: `adv1_proj_nba_3p` kept over raw `pg_3p_pct` — Tankathon's
projection incorporates FT%, volume, and age. Better signal for NBA translation.

---

## PCA Component Interpretations (from actual loadings)

Verified against real pca.pkl loadings. Use these for the Gemini prompt.

| Component | Variance | Interpretation | Key Positive Loadings | Key Negative Loadings |
|---|---|---|---|---|
| PC1 | 26.5% | Size & Interior Dominance | Height, Reb/36, Blk/36, Weight | 3PA Rate |
| PC2 | 14.6% | Offensive Star Power | OWS/40, OBPM, PER, Pts/36, TS% | (one-sided — low = role player) |
| PC3 | 8.2% | Two-Way Playmaking | AST/USG, Ast/36, Stl/36, DBPM | (off-ball specialist if negative) |
| PC4 | 6.8% | Usage & Volume vs Efficiency | USG%, TO/36, Pts/36 | DBPM, TS% |
| PC5 | 5.8% | Athletic Explosiveness | Vertical Jump | 3/4 Sprint, Lane Agility* |
| PC6 | 4.3% | Length & Maturity | Wingspan/Height ratio, MPG, Age | Fouls/36 |
| PC9 | 3.1% | Durability | Games Played (0.773 loading) | — |
| PC7,8,10-15 | — | Noise — ignore in analysis | — | — |

*For PC5: lower sprint/agility times = more explosive (faster = better athlete)
so negative loadings on sprint = positive athleticism signal.

PC5 and combine data: only reference for players where limited_data=false.

---

## Animal Archetype System (NEW — Central Feature)

### Concept
Every player in the universe has an animal archetype generated by Gemini
based on their PCA scores + raw stats. This is the primary identity shown
in player bubbles — more prominent than name/position.

### Two flows:
1. **Existing players (462)** — PRECOMPUTED. Run batch script once using
   ChatGPT or Gemini, store in `player_archetypes.json`. No API calls at
   runtime. Served statically from backend.

2. **New prospect input** — LIVE Gemini call. Streams into the prospect
   bubble in real time as the camera animation finishes. Writeup ends with
   one factual sentence naming the 3 comps:
   "His closest historical comparisons are [Comp1], [Comp2], and [Comp3]."
   This sentence is included in the Gemini prompt so it writes it naturally
   as the final sentence of the writeup.

### Output format (strict — must be parseable):
```
ANIMAL: [ANIMAL NAME IN CAPS]
ARCHETYPE: [3-5 Word Label In Title Case]
WRITEUP: [3-4 sentences, under 90 words]
```

### Bubble display order:
1. Animal name (large, prominent)
2. Archetype label (smaller, below animal)
3. Key stats (pts/reb/ast or relevant stats)
4. Writeup text (streams for prospect, static for existing players)

### player_archetypes.json structure:
```json
{
  "cooper-flagg": {
    "animal": "WOLVERINE",
    "archetype": "Relentless Two-Way Wing",
    "writeup": "This player is a wolverine..."
  },
  "john-wall": { ... }
}
```
Key = slug derived from name column (player_slug not in players_umap_3d.csv — dropped during pipeline).
Derivation: `"John Wall PG | Kentucky"` → strip school → strip position → lowercase → hyphenate:
```python
def name_to_slug(raw):
    clean = raw.split("|")[0].strip()   # "John Wall PG"
    name_only = clean.rsplit(" ", 1)[0] # "John Wall"
    return name_only.lower().replace(" ", "-").replace("'", "")
```
NOTE: Add player_slug back to ID_COLS in prepare_data.py for future runs.

### Prompt file: scouting_prompt.py
Contains `GEMINI_PROMPT_TEMPLATE` and `build_scouting_prompt(player)` function.
Uses real PCA loadings with variance percentages. Handles limited_data flag.
~1,255 tokens per call. Test with ChatGPT before batch processing all 462.

---

## Similarity Search

**CRITICAL: PCA space only, NEVER UMAP space.**
UMAP distorts distances for visualization only.

```python
def find_similar(query_pca_vector, all_pca_matrix, n=3, skip_first=False):
    dists = np.linalg.norm(all_pca_matrix - query_pca_vector, axis=1)
    top_idx = np.argsort(dists)
    if skip_first:
        top_idx = top_idx[1:n+1]  # skip self (existing player click flow)
    else:
        top_idx = top_idx[:n]     # no self to skip (new prospect flow)
    return top_idx, dists[top_idx]
```
NOTE: Use skip_first=True when finding neighbors for an existing player
(their own vector IS in the matrix). Use skip_first=False for new prospects.

### Transforming a new prospect:
```python
scaler  = pickle.load(open("scaler.pkl", "rb"))
pca     = pickle.load(open("pca.pkl", "rb"))
reducer = pickle.load(open("umap_model.pkl", "rb"))
# Missing values imputed with training medians BEFORE scaling
scaled      = scaler.transform(prospect_vector)  # (1, 32)
pca_vec     = pca.transform(scaled)              # (1, 15) — similarity
umap_coords = reducer.transform(pca_vec)         # (1, 3)  — visualization
```

---

## API Endpoints (FINAL — 4 total)

| Method | Path | Description |
|---|---|---|
| GET | `/players` | All 462 players: umap_x/y/z, name, position, draft_year, pick, limited_data, key stats, archetype |
| POST | `/similarity` | Prospect stats → 3 nearest neighbors + prospect umap coords |
| POST | `/similarity/blurb` | Prospect stats → streaming Gemini archetype+writeup (SSE) |
| GET | `/players/{slug}/similar` | Existing player slug → their 3 nearest neighbors (skip_first=True) |

Notes:
- `/players` response includes precomputed archetype (animal+label+writeup)
  for each player, loaded from player_archetypes.json at startup
- Medians included in `/players` response metadata for form placeholders
- `/similarity` and `/similarity/blurb` called in parallel on form submit
- `/players/{slug}/similar` used for click-to-explore flow — finds neighbors
  server-side using skip_first=True since player IS in the PCA matrix
- slug derived via name_to_slug() to match player_archetypes.json keys

### POST /similarity request body
```json
{
  "position": "SF",
  "height_inches": 81.0,
  "weight": 210.0,
  "age_at_draft": 19.5,
  "p36_pts": 22.5,
  "n_results": 3
}
```
Any missing fields imputed server-side with stored training medians.

---

## Three.js Scene (CURRENT STATE — working)

- **Renderer**: `antialias: false` (keeps pixel edges sharp), ACESFilmic tone
  mapping, exposure 1.1
- **Post-processing**: UnrealBloomPass (strength 1.2, radius 0.3, threshold
  0.55) — tight phosphor-CRT bloom on bright pixel centers only
- **Background**: pure `#000000`, no fog, no nebula sphere
- **Player sprites**: 462 THREE.Sprite, pixel-art 4-point stars, 5 blue/cyan
  variants, scale 7–14, AdditiveBlending. Z-axis scaled ×150 vs ×100 for X/Y
  (compensates for UMAP z-axis flatness)
- **Background points**: 3000 THREE.Points, color #aaccff, size 0.9, opacity 0.45
- **Ambient blink**: every field star has a random period (1.8–5s) and phase;
  square-wave opacity flip in animate loop
- **Navigation**: WASD + mouse-drag look + scroll-wheel glide. Idle cinematic
  drift (Lissajous path) before form submit. Camera fly-in via GSAP
  CatmullRomCurve3 on prospect submit, then WASD unlocks.
- **Camera bug fix**: final camera position derived from centroid−prospect
  direction so prospect is always in frame regardless of UMAP position

### Prospect markers (added after similarity search):
- 8-point gold pixel-art star sprite, scale 26→32 snap pulse, slow rotation
- 160-particle gold ring (THREE.Points) orbiting the prospect position

### Comp markers (added after fly-in completes):
- Existing player sprites swapped to orange (#ff7700) pixel-art texture
  with white outline, scale 18
- Blink overridden to 0.45s period, in-sync, full on/off

---

## Backend Files (CURRENT STATE — working)

### main.py highlights:
- lifespan context manager calls ml.load_all() on startup
- CORS: allow_credentials=False (wildcard origins ok without credentials)
- ProspectInput pydantic model — all fields optional except n_results
- _prepare_features() computes wingspan_over_height before passing to ml
- /similarity/blurb checks GEMINI_API_KEY at request time, not startup

### ml.py highlights:
- _State class holds df, pca_matrix, scaler, pca, reducer, medians
- load_all() called once at startup
- find_similar(query, n, skip_first=False) — Euclidean distance in PCA space
  skip_first=True for existing players (self is in matrix)
  skip_first=False for new prospects (self not in matrix)
- transform_prospect() — imputes NaNs with medians, scales, PCA, UMAP
- player_to_dict() — NaN→None for clean JSON, includes limited_data flag
- height_display() — converts decimal inches to ft/in string

---

## Project Structure

```
project/
├── backend/
│   ├── main.py
│   ├── ml.py
│   ├── requirements.txt
│   └── data/
│       ├── players_umap_3d.csv
│       ├── player_archetypes.json    ← precomputed, generate with batch script
│       ├── scaler.pkl
│       ├── pca.pkl
│       └── umap_model.pkl
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Universe.jsx          ← Three.js scene (working)
│   │   │   ├── ProspectForm.jsx      ← input form (working)
│   │   │   ├── PlayerBubble.jsx      ← hover/click bubble (not built yet)
│   │   │   └── GeminiBubble.jsx      ← streaming writeup bubble (not built yet)
│   │   └── App.jsx
│   └── package.json
└── data/
    ├── scouting_prompt.py            ← Gemini prompt builder
    ├── players_umap_3d.csv
    └── [other data files]
```

---

## Verified Similarity Results (top 3)

| Query | Top 3 comps | Verdict |
|---|---|---|
| Cooper Flagg | Jarrett Culver, Jabari Smith, Otto Porter | ✅ |
| Zion Williamson | Okongwu, Obi Toppin, Ayton | ✅ |
| John Wall | SGA, De'Aaron Fox, Dennis Smith | ✅ excellent |
| KAT | Embiid, Chet Holmgren, Zach Collins | ✅ |
| Stephon Castle | Maxey, Justise Winslow, Keldon Johnson | ✅ |
| Wembanyama | Swanigan, Banchero, Bennett | ⚠️ imputation effect — limited_data |

---

## Name Format Note

CSV `name` field: `"Cooper Flagg SF/PF | Duke"`
Clean name: `name.split("|")[0].strip().rsplit(" ", 2)[0]`
School: `name.split("|")[1].strip()`
player_slug: NOT in players_umap_3d.csv — derive using name_to_slug() in scouting_prompt.py
  e.g. "John Wall PG | Kentucky" → "john-wall"

---

## Outstanding TODOs (in order)

1. **Camera animation** — GSAP snake through universe, land on prospect
2. **Player bubbles** — PlayerBubble.jsx with animal/archetype/stats/writeup
3. **Gemini streaming bubble** — GeminiBubble.jsx, prospect only, streams live
4. **Click to explore** — clicking any star calls GET /players/{slug}/similar,
   shows precomputed archetype bubble + their 3 nearest neighbors illuminate
5. **Search again button** — unobtrusive, returns to form
6. **Batch precompute archetypes** — run scouting_prompt.py on all 462 players
   using ChatGPT, save to player_archetypes.json
7. **Update /players endpoint** — include archetype from player_archetypes.json
8. **Add player_slug to ID_COLS in prepare_data.py** — for future pipeline runs
9. **Deploy** — Railway (backend) + Vercel (frontend)