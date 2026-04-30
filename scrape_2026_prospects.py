"""
2026 NBA Draft Prospect Scraper
Scrapes current prospects from tankathon.com/mock_draft,
visits each player page, outputs prospects_2026.json for the frontend form.

Usage:
    pip install requests beautifulsoup4
    python scrape_2026_prospects.py

Output: prospects_2026.json — array of prospect objects ready for the form
"""

import csv, re, time
from bs4 import BeautifulSoup, NavigableString, Tag
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.tankathon.com"
DELAY    = 1.5
OUTPUT   = "prospects_2026.csv"

_playwright = None
_browser = None
_page = None

def _get_page():
    global _playwright, _browser, _page
    if _page is None:
        _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(headless=True)
        _page = _browser.new_page(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    return _page

def get_soup(url):
    try:
        page = _get_page()
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        # Scroll down to trigger lazy-loaded stats
        page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        page.wait_for_timeout(500)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1500)
        html = page.content()
        return BeautifulSoup(html, "html.parser")
    except Exception as e:
        print(f"  [WARN] {url}: {e}"); return None

def cleanup():
    global _playwright, _browser, _page
    if _browser: _browser.close()
    if _playwright: _playwright.stop()

def t(el): return el.get_text(separator=" ", strip=True) if el else ""

def get_tokens(soup):
    SKIP = {"script","style","nav","head","meta","link","noscript","footer","svg","img","button"}
    # stat-label elements (e.g. "PF", "TO") can appear multiple times legitimately
    # (once as position label, once as stat label) so they are NOT deduplicated.
    # All other elements use deduplication to avoid noise.
    STAT_CLASSES = {"stat-label", "stat-data"}
    seen, raw = set(), []
    for el in soup.descendants:
        if not isinstance(el, Tag): continue
        if el.name in SKIP: continue
        el_classes = set(el.get("class") or [])
        is_stat = bool(el_classes & STAT_CLASSES)
        direct = "".join(str(c).strip() for c in el.children
                         if isinstance(c, NavigableString) and str(c).strip())
        if direct:
            if is_stat or direct not in seen:
                if not is_stat: seen.add(direct)
                raw.append(direct)
        if not el.find(True):
            full = el.get_text(strip=True)
            if full:
                if is_stat or full not in seen:
                    if not is_stat: seen.add(full)
                    raw.append(full)
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
    return bool(re.match(r"^[\d.]", tok)) and "/" not in tok

def parse_height(s):
    if not s: return None
    s = s.strip().replace("\u2019", "'")
    m = re.match(r"(\d+)'([\d.]+)\"?", s)
    if m: return float(m.group(1)) * 12 + float(m.group(2))
    try: return float(s)
    except: return None

def parse_weight(s):
    if not s: return None
    try: return float(s.replace("lbs","").strip())
    except: return None

def parse_age(s):
    if not s: return None
    try: return float(s.replace("yrs","").strip())
    except: return None

def parse_stat(s):
    if not s: return None
    try: return float(s)
    except: return None

# ── Scrape mock draft page for slugs ──────────────────────────────────────────
# Known non-player slugs to skip (nav links, tool pages)
SKIP_SLUGS = {"compare"}

def scrape_mock_draft_slugs():
    print("Fetching mock draft page...")
    soup = get_soup(f"{BASE_URL}/mock_draft")
    if not soup: return []
    slugs = []
    seen = set()
    for node in soup.descendants:
        if len(slugs) >= 30:
            break
        if not isinstance(node, Tag) or node.name != "a":
            continue
        href = node.get("href", "")
        if not re.match(r"^/players/", href):
            continue
        slug = href.split("/players/")[-1].rstrip("/")
        # Skip non-player slugs and slugs with slashes (sub-pages)
        if not slug or slug in SKIP_SLUGS or "/" in slug:
            continue
        # Skip if link text looks like a tool name not a player
        name = t(node).split("\n")[0].strip()
        if not name or len(name) < 3 or name.lower() in {"compare", "prospects", "players"}:
            continue
        if slug not in seen:
            seen.add(slug)
            slugs.append({"slug": slug, "name": name})
    print(f"  Found {len(slugs)} prospects (capped at 30)")
    return slugs

# ── Scrape individual player page ─────────────────────────────────────────────
SECTION_HEADERS = {
    "nba combine":       "combine",
    "per game averages": "per_game",
    "per 36 minutes":    "per_36",
    "advanced stats ii": "adv2",
    "advanced stats i":  "adv1",
}
COMBINE_SETS = {
    frozenset(["max","vertical"]):   "combine_max_vertical",
    frozenset(["lane agility"]):     "combine_lane_agility",
    frozenset(["lane","agility"]):   "combine_lane_agility",
    frozenset(["shuttle"]):          "combine_shuttle",
    frozenset(["3/4","sprint"]):     "combine_three_qtr_sprint",
    frozenset(["standing","reach"]): "combine_standing_reach",
    frozenset(["wingspan"]):         "combine_wingspan_inches",
}
PG_MAP = {
    "g":"pg_g","mp":"pg_mp","fgm-fga":"pg_fgm_fga","fg%":"pg_fg_pct",
    "3pm-3pa":"pg_tpm_tpa","3p%":"pg_3p_pct","ftm-fta":"pg_ftm_fta","ft%":"pg_ft_pct",
    "reb":"pg_reb","ast":"pg_ast","blk":"pg_blk","stl":"pg_stl",
    "to":"pg_to","pf":"pg_pf","pts":"pg_pts"
}
ADV1_MAP = {
    "ts%":"adv1_ts_pct","efg%":"adv1_efg_pct","3par":"adv1_3pa_rate",
    "ftar":"adv1_fta_rate","nba 3p%":"adv1_proj_nba_3p","usg%":"adv1_usg_pct",
    "ast/usg":"adv1_ast_usg","ast/to":"adv1_ast_to"
}
ADV2_MAP = {
    "per":"adv2_per","ows/40":"adv2_ows_40","dws/40":"adv2_dws_40","ws/40":"adv2_ws_40",
    "ortg":"adv2_ortg","drtg":"adv2_drtg","obpm":"adv2_obpm","dbpm":"adv2_dbpm","bpm":"adv2_bpm"
}
SECTION_FIELD_MAPS = {"per_game": PG_MAP, "adv1": ADV1_MAP, "adv2": ADV2_MAP}
STOP = {"stat strengths","stat weaknesses","game log |","determining the nba draft order"}

def normalise(s):
    s = re.sub(r"^\d{4}-\d{2}\s+", "", s)
    s = re.sub(r"\s*(hover|tap).*$", "", s, flags=re.I)
    return s.strip().lower()

def scrape_player(slug):
    soup = get_soup(f"{BASE_URL}/players/{slug}")
    if not soup: return {}
    tokens = get_tokens(soup)
    data = {}

    # Bio — extract position directly from class="position" div (most reliable)
    pos_el = soup.find(class_="position")
    if pos_el:
        pos_text = pos_el.get_text(strip=True)
        if re.fullmatch(r"(PG|SG|SF|PF|C|PG/SG|SG/SF|SF/PF|PF/C|SG/PF|PG/SF|SF/SG|C/PF|PF/SF|SG/PG|PF/SF)", pos_text):
            data["position"] = pos_text

    # Bio — extract directly from class="measurable" divs
    # Structure: <div class="measurable">
    #              <div class="label">Height</div>
    #              <div class="value"><span class="feet">6'</span><span class="inches">9"</span></div>
    #            </div>
    for m in soup.find_all(class_="measurable"):
        label_el = m.find(class_="label")
        value_el = m.find(class_="value")
        if not label_el or not value_el: continue
        label = label_el.get_text(strip=True).lower()
        if label == "height":
            feet_el = value_el.find(class_="feet")
            inches_el = value_el.find(class_="inches")
            if feet_el and inches_el:
                data["height"] = feet_el.get_text(strip=True) + inches_el.get_text(strip=True)
            else:
                data["height"] = value_el.get_text(strip=True)
        elif label == "weight":
            data["weight"] = value_el.get_text(strip=True)
        elif "'24 draft age" in label or "draft age" in label or label == "age":
            val = value_el.get_text(strip=True)
            data["age_at_draft"] = val if val.endswith("yrs") else val + "yrs"

    # Fallback: scan for age pattern in full text
    if not data.get("age_at_draft"):
        for el in soup.find_all(string=re.compile(r"\d+\.\d+yrs")):
            data["age_at_draft"] = el.strip(); break

    current_section = None; field_map = {}; pending_label = None; combine_parts = []
    for tok in tokens:
        low = tok.lower()
        if any(s in low for s in STOP): break
        n = normalise(tok)
        sec = next((key for hdr, key in SECTION_HEADERS.items() if hdr in n), None)
        if sec:
            current_section = sec
            field_map = SECTION_FIELD_MAPS.get(sec, {})
            pending_label = None; combine_parts = []
            continue
        if current_section is None: continue
        if current_section == "combine":
            if is_combine_value(tok):
                key = frozenset(p.lower() for p in combine_parts)
                dest = COMBINE_SETS.get(key)
                if dest: data[dest] = tok
                combine_parts = []
            else:
                if frozenset(p.lower() for p in combine_parts) in {frozenset(["shuttle"]),frozenset(["wingspan"])}:
                    combine_parts = []
                combine_parts.append(tok.lower())
            continue
        if current_section == "per_36": continue
        low_tok = low.rstrip(":")
        if low_tok in field_map:
            pending_label = low_tok
        elif pending_label is not None:
            # Only store value if it looks numeric — prevents label-shift
            # when a stat is missing from the page (e.g. TO not rendered)
            if re.match(r"^[\d.]", tok) or tok.startswith("-"):
                data[field_map[pending_label]] = tok
            pending_label = None

    # Calculate per-36 from per-game
    def p36(v, mp):
        try: return round(float(v)*36/float(mp), 1)
        except: return None
    def p36s(s, mp):
        try:
            m, a = s.split("-"); f = 36/float(mp)
            return f"{round(float(m)*f,1)}-{round(float(a)*f,1)}"
        except: return None
    mp = data.get("pg_mp")
    if mp:
        data["p36_g"]       = data.get("pg_g")
        data["p36_mp"]      = 36.0
        data["p36_fgm_fga"] = p36s(data.get("pg_fgm_fga",""), mp)
        data["p36_fg_pct"]  = parse_stat(data.get("pg_fg_pct"))
        data["p36_tpm_tpa"] = p36s(data.get("pg_tpm_tpa",""), mp)
        data["p36_3p_pct"]  = parse_stat(data.get("pg_3p_pct"))
        data["p36_ftm_fta"] = p36s(data.get("pg_ftm_fta",""), mp)
        data["p36_ft_pct"]  = parse_stat(data.get("pg_ft_pct"))
        data["p36_reb"]     = p36(data.get("pg_reb"), mp)
        data["p36_ast"]     = p36(data.get("pg_ast"), mp)
        data["p36_blk"]     = p36(data.get("pg_blk"), mp)
        data["p36_stl"]     = p36(data.get("pg_stl"), mp)
        data["p36_to"]      = p36(data.get("pg_to"), mp)
        data["p36_pf"]      = p36(data.get("pg_pf"), mp)
        data["p36_pts"]     = p36(data.get("pg_pts"), mp)

    return data

# ── Convert raw scraped data to form-ready format ─────────────────────────────
def to_form_data(slug, name, raw):
    """Convert raw scraped strings to typed values for the frontend form."""
    return {
        "slug": slug,
        "name": name.split("|")[0].strip().rsplit(" ", 1)[0] if "|" in name else name,
        "school": name.split("|")[1].strip() if "|" in name else "",
        "position": raw.get("position", ""),
        "height_inches": parse_height(raw.get("height")),
        "weight": parse_weight(raw.get("weight")),
        "age_at_draft": parse_age(raw.get("age_at_draft")),
        # Combine (likely null — combine hasn't happened yet)
        "combine_max_vertical": parse_stat(raw.get("combine_max_vertical")),
        "combine_lane_agility": parse_stat(raw.get("combine_lane_agility")),
        "combine_shuttle": parse_stat(raw.get("combine_shuttle")),
        "combine_three_qtr_sprint": parse_stat(raw.get("combine_three_qtr_sprint")),
        "combine_wingspan_inches": parse_height(raw.get("combine_wingspan_inches")),
        # Per game
        "pg_g": parse_stat(raw.get("pg_g")),
        "pg_mp": parse_stat(raw.get("pg_mp")),
        "pg_fg_pct": parse_stat(raw.get("pg_fg_pct")),
        "pg_ft_pct": parse_stat(raw.get("pg_ft_pct")),
        "pg_to": parse_stat(raw.get("pg_to")),
        "pg_pts": parse_stat(raw.get("pg_pts")),
        "pg_reb": parse_stat(raw.get("pg_reb")),
        "pg_ast": parse_stat(raw.get("pg_ast")),
        "pg_blk": parse_stat(raw.get("pg_blk")),
        "pg_stl": parse_stat(raw.get("pg_stl")),
        "pg_pf": parse_stat(raw.get("pg_pf")),
        # Per 36
        "p36_pts": raw.get("p36_pts"),
        "p36_reb": raw.get("p36_reb"),
        "p36_ast": raw.get("p36_ast"),
        "p36_blk": raw.get("p36_blk"),
        "p36_stl": raw.get("p36_stl"),
        "p36_to":  raw.get("p36_to"),
        "p36_pf":  raw.get("p36_pf"),
        # Advanced I
        "adv1_ts_pct":      parse_stat(raw.get("adv1_ts_pct")),
        "adv1_3pa_rate":    parse_stat(raw.get("adv1_3pa_rate")),
        "adv1_fta_rate":    parse_stat(raw.get("adv1_fta_rate")),
        "adv1_proj_nba_3p": parse_stat(raw.get("adv1_proj_nba_3p")),
        "adv1_usg_pct":     parse_stat(raw.get("adv1_usg_pct")),
        "adv1_ast_usg":     parse_stat(raw.get("adv1_ast_usg")),
        "adv1_ast_to":      parse_stat(raw.get("adv1_ast_to")),
        # Advanced II
        "adv2_per":  parse_stat(raw.get("adv2_per")),
        "adv2_ows_40": parse_stat(raw.get("adv2_ows_40")),
        "adv2_dws_40": parse_stat(raw.get("adv2_dws_40")),
        "adv2_obpm": parse_stat(raw.get("adv2_obpm")),
        "adv2_dbpm": parse_stat(raw.get("adv2_dbpm")),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    slugs = scrape_mock_draft_slugs()
    if not slugs:
        print("No prospects found."); return

    prospects = []
    for i, p in enumerate(slugs):
        print(f"  [{i+1:>2}/{len(slugs)}] {p['name']}")
        raw = scrape_player(p["slug"])
        time.sleep(DELAY)
        form_data = to_form_data(p["slug"], p["name"], raw)
        prospects.append(form_data)

    import csv
    if prospects:
        with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=prospects[0].keys())
            w.writeheader()
            w.writerows(prospects)

    print(f"\nDone! {len(prospects)} prospects -> {OUTPUT}")
    cleanup()

    # Quick sanity check
    print("\nSample (first 3):")
    for p in prospects[:3]:
        print(f"  {p['name']:<25} pos={p['position']:<6} "
              f"ht={p['height_inches']} wt={p['weight']} "
              f"pts/36={p['p36_pts']} obpm={p['adv2_obpm']}")

if __name__ == "__main__":
    main()