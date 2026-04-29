"""
Tankathon Past Draft Scraper
Scrapes 1st-round picks 2010-2025, visits each player page, outputs CSV.
    pip install requests beautifulsoup4
    python tankathon_scraper.py
"""
import csv, os, re, sys, time, requests
from dataclasses import dataclass, fields, asdict
from bs4 import BeautifulSoup, NavigableString, Tag

BASE_URL    = "https://www.tankathon.com"
YEARS       = list(range(2010, 2026))
ROUND_LIMIT = 30
DELAY       = 1.5
OUTPUT_CSV  = "tankathon_draft_picks.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

@dataclass
class Player:
    draft_year: str = ""; pick: str = ""; name: str = ""; player_slug: str = ""; player_url: str = ""
    position: str = ""; height: str = ""; weight: str = ""; age_at_draft: str = ""
    combine_max_vertical: str = ""; combine_lane_agility: str = ""; combine_shuttle: str = ""
    combine_three_qtr_sprint: str = ""; combine_standing_reach: str = ""; combine_wingspan: str = ""
    pg_g: str = ""; pg_mp: str = ""; pg_fgm_fga: str = ""; pg_fg_pct: str = ""
    pg_tpm_tpa: str = ""; pg_3p_pct: str = ""; pg_ftm_fta: str = ""; pg_ft_pct: str = ""
    pg_reb: str = ""; pg_ast: str = ""; pg_blk: str = ""; pg_stl: str = ""
    pg_to: str = ""; pg_pf: str = ""; pg_pts: str = ""
    p36_g: str = ""; p36_mp: str = ""; p36_fgm_fga: str = ""; p36_fg_pct: str = ""
    p36_tpm_tpa: str = ""; p36_3p_pct: str = ""; p36_ftm_fta: str = ""; p36_ft_pct: str = ""
    p36_reb: str = ""; p36_ast: str = ""; p36_blk: str = ""; p36_stl: str = ""
    p36_to: str = ""; p36_pf: str = ""; p36_pts: str = ""
    adv1_ts_pct: str = ""; adv1_efg_pct: str = ""; adv1_3pa_rate: str = ""
    adv1_fta_rate: str = ""; adv1_proj_nba_3p: str = ""; adv1_usg_pct: str = ""
    adv1_ast_usg: str = ""; adv1_ast_to: str = ""
    adv2_per: str = ""; adv2_ows_40: str = ""; adv2_dws_40: str = ""; adv2_ws_40: str = ""
    adv2_ortg: str = ""; adv2_drtg: str = ""; adv2_obpm: str = ""; adv2_dbpm: str = ""; adv2_bpm: str = ""

session = requests.Session()
session.headers.update(HEADERS)

def get_soup(url):
    try:
        r = session.get(url, timeout=15); r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"  [WARN] {url}: {e}", file=sys.stderr); return None

def t(el):
    return el.get_text(separator=" ", strip=True) if el else ""

def get_tokens(soup):
    SKIP = {"script","style","nav","head","meta","link","noscript","footer","svg","img","button"}
    seen, raw = set(), []
    for el in soup.descendants:
        if not isinstance(el, Tag): continue
        if el.name in SKIP: continue
        direct = "".join(str(c).strip() for c in el.children
                         if isinstance(c, NavigableString) and str(c).strip())
        if direct and direct not in seen:
            seen.add(direct); raw.append(direct)
        if not el.find(True):
            full = el.get_text(strip=True)
            if full and full not in seen:
                seen.add(full); raw.append(full)
    merged, i = [], 0
    while i < len(raw):
        tok = raw[i]
        if i + 1 < len(raw):
            nxt = raw[i+1]
            if re.fullmatch(r"\d+['\u2019]", tok) and re.fullmatch(r'[\d.]+\"', nxt):
                merged.append(tok+nxt); i+=2; continue
            if re.fullmatch(r"\d+", tok) and nxt == "lbs":
                merged.append(tok+nxt); i+=2; continue
            if re.fullmatch(r"\d+\.\d+", tok) and nxt == "yrs":
                merged.append(tok+nxt); i+=2; continue
        merged.append(tok); i+=1
    return merged

def is_combine_value(tok):
    """True if token is a numeric measurement (not a label like '3/4')."""
    return bool(re.match(r"^[\d.]", tok)) and "/" not in tok

def scrape_draft_year(year):
    url = f"{BASE_URL}/past_drafts/{year}"
    print(f"\n[{year}] {url}")
    soup = get_soup(url)
    if not soup: return []
    picks = []
    seen_picks = set()
    pending_ints = []
    in_round1 = True
    for node in soup.descendants:
        if isinstance(node, NavigableString):
            txt = node.strip()
            if re.search(r"Round\s*2", txt, re.I): in_round1 = False
            if not in_round1: continue
            if re.fullmatch(r"\d+", txt): pending_ints.append(int(txt))
        elif isinstance(node, Tag) and node.name == "a":
            if not in_round1: continue
            href = node.get("href", "")
            if "/players/" in href:
                slug = href.split("/players/")[-1].rstrip("/")
                name = t(node).split("\n")[0].strip()
                valid = [n for n in pending_ints if 1 <= n <= ROUND_LIMIT]
                pick_num = valid[0] if valid else None
                if pick_num and pick_num not in seen_picks:
                    seen_picks.add(pick_num)
                    picks.append({"draft_year": str(year), "pick": str(pick_num),
                                  "name": name, "player_slug": slug})
                pending_ints = []
    picks.sort(key=lambda p: int(p["pick"]))
    print(f"  -> {len(picks)} picks")
    return picks

SECTION_HEADERS = {
    "nba combine":       "combine",
    "per game averages": "per_game",
    "per 36 minutes":    "per_36",
    "advanced stats ii": "adv2",   # ii before i
    "advanced stats i":  "adv1",
}
COMBINE_SETS = {
    frozenset(["max","vertical"]):   "combine_max_vertical",
    frozenset(["lane agility"]):     "combine_lane_agility",
    frozenset(["lane","agility"]):   "combine_lane_agility",
    frozenset(["shuttle"]):          "combine_shuttle",
    frozenset(["3/4","sprint"]):     "combine_three_qtr_sprint",
    frozenset(["standing","reach"]): "combine_standing_reach",
    frozenset(["wingspan"]):         "combine_wingspan",
}
PG_MAP   = {"g":"pg_g","mp":"pg_mp","fgm-fga":"pg_fgm_fga","fg%":"pg_fg_pct",
            "3pm-3pa":"pg_tpm_tpa","3p%":"pg_3p_pct","ftm-fta":"pg_ftm_fta","ft%":"pg_ft_pct",
            "reb":"pg_reb","ast":"pg_ast","blk":"pg_blk","stl":"pg_stl",
            "to":"pg_to","pf":"pg_pf","pts":"pg_pts"}
ADV1_MAP = {"ts%":"adv1_ts_pct","efg%":"adv1_efg_pct","3par":"adv1_3pa_rate",
            "ftar":"adv1_fta_rate","nba 3p%":"adv1_proj_nba_3p","usg%":"adv1_usg_pct",
            "ast/usg":"adv1_ast_usg","ast/to":"adv1_ast_to"}
ADV2_MAP = {"per":"adv2_per","ows/40":"adv2_ows_40","dws/40":"adv2_dws_40","ws/40":"adv2_ws_40",
            "ortg":"adv2_ortg","drtg":"adv2_drtg","obpm":"adv2_obpm","dbpm":"adv2_dbpm","bpm":"adv2_bpm"}
SECTION_FIELD_MAPS = {"per_game": PG_MAP, "adv1": ADV1_MAP, "adv2": ADV2_MAP}
STOP = {"stat strengths","stat weaknesses","game log |","determining the nba draft order"}

def normalise(s):
    s = re.sub(r"^\d{4}-\d{2}\s+", "", s)
    s = re.sub(r"\s*(hover|tap).*$", "", s, flags=re.I)
    return s.strip().lower()

def scrape_player(slug):
    url = f"{BASE_URL}/players/{slug}"
    soup = get_soup(url)
    if not soup: return {}
    tokens = get_tokens(soup)
    data = {"player_url": url}

    # Bio
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low == "height" and i+1 < len(tokens): data["height"] = tokens[i+1]
        elif low == "weight" and i+1 < len(tokens): data["weight"] = tokens[i+1]
        elif re.search(r"\d+\.\d+yrs$", tok): data["age_at_draft"] = tok
        elif re.fullmatch(r"(PG|SG|SF|PF|C|PG/SG|SG/SF|SF/PF|PF/C|SG/PF|PG/SF|SF/SG|C/PF|PF/SF)", tok):
            if not data.get("position"): data["position"] = tok

    # Stats
    current_section = None
    field_map = {}
    pending_label = None
    combine_parts = []

    for tok in tokens:
        low = tok.lower()
        if any(s in low for s in STOP): break
        n = normalise(tok)
        sec = next((key for hdr, key in SECTION_HEADERS.items() if hdr in n), None)
        if sec:
            current_section = sec
            field_map = SECTION_FIELD_MAPS.get(sec, {})
            pending_label = None
            combine_parts = []
            continue
        if current_section is None: continue

        # Combine: accumulate label words, store when numeric value found
        if current_section == "combine":
            if is_combine_value(tok):
                key = frozenset(p.lower() for p in combine_parts)
                dest = COMBINE_SETS.get(key)
                if dest: data[dest] = tok
                combine_parts = []
            else:
                # Discard single-word label if it has no value (e.g. shuttle not recorded)
                if frozenset(p.lower() for p in combine_parts) in {frozenset(["shuttle"]), frozenset(["wingspan"])}:
                    combine_parts = []
                combine_parts.append(tok.lower())
            continue

        # Per-36: skip — calculated from per-game below
        if current_section == "per_36": continue

        # Per-game, Adv1, Adv2: label then value
        low_tok = low.rstrip(":")
        if low_tok in field_map:
            pending_label = low_tok
        elif pending_label is not None:
            data[field_map[pending_label]] = tok
            pending_label = None

    # Calculate per-36 from per-game (accurate, no positional parsing issues)
    def p36(v, mp):
        try: return str(round(float(v) * 36 / float(mp), 1))
        except: return ""
    def p36s(s, mp):
        try:
            m, a = s.split("-"); f = 36 / float(mp)
            return f"{round(float(m)*f,1)}-{round(float(a)*f,1)}"
        except: return ""
    mp = data.get("pg_mp", "")
    if mp:
        data["p36_g"]       = data.get("pg_g", "")
        data["p36_mp"]      = "36.0"
        data["p36_fgm_fga"] = p36s(data.get("pg_fgm_fga", ""), mp)
        data["p36_fg_pct"]  = data.get("pg_fg_pct", "")
        data["p36_tpm_tpa"] = p36s(data.get("pg_tpm_tpa", ""), mp)
        data["p36_3p_pct"]  = data.get("pg_3p_pct", "")
        data["p36_ftm_fta"] = p36s(data.get("pg_ftm_fta", ""), mp)
        data["p36_ft_pct"]  = data.get("pg_ft_pct", "")
        data["p36_reb"]     = p36(data.get("pg_reb", ""), mp)
        data["p36_ast"]     = p36(data.get("pg_ast", ""), mp)
        data["p36_blk"]     = p36(data.get("pg_blk", ""), mp)
        data["p36_stl"]     = p36(data.get("pg_stl", ""), mp)
        data["p36_to"]      = p36(data.get("pg_to", ""), mp)
        data["p36_pf"]      = p36(data.get("pg_pf", ""), mp)
        data["p36_pts"]     = p36(data.get("pg_pts", ""), mp)

    return data

def main():
    all_players = []
    valid_fields = {f.name for f in fields(Player)}
    for year in YEARS:
        picks = scrape_draft_year(year)
        time.sleep(DELAY)
        for p in picks:
            print(f"  [{p['pick']:>2}] {p['name']}")
            profile = scrape_player(p["player_slug"])
            time.sleep(DELAY)
            merged = {**p, **profile}
            all_players.append(Player(**{k: v for k, v in merged.items() if k in valid_fields}))
    if not all_players:
        print("\n[WARN] No data."); return
    tmp = OUTPUT_CSV + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(Player)])
        w.writeheader()
        for p in all_players: w.writerow(asdict(p))
    try:
        os.replace(tmp, OUTPUT_CSV)
        print(f"\nDone! {len(all_players)} players -> {OUTPUT_CSV}")
    except PermissionError:
        print(f"\n[WARN] {OUTPUT_CSV} is open. Data saved to {tmp} — close Excel and rename it.")

if __name__ == "__main__":
    main()