from __future__ import annotations

import re
import unicodedata
import pandas as pd
import requests
from pathlib import Path
from scrapy.http import TextResponse

# Daily Faceoff's team pages use these slugs.
TEAM_SLUGS = {
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres",
    "CGY": "calgary-flames", "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche", "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars",
    "DET": "detroit-red-wings", "EDM": "edmonton-oilers", "FLA": "florida-panthers",
    "LAK": "los-angeles-kings", "MIN": "minnesota-wild", "MTL": "montreal-canadiens",
    "NSH": "nashville-predators", "NJD": "new-jersey-devils", "NYI": "new-york-islanders",
    "NYR": "new-york-rangers", "OTT": "ottawa-senators", "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks", "SEA": "seattle-kraken",
    "STL": "st-louis-blues", "TBL": "tampa-bay-lightning", "TOR": "toronto-maple-leafs",
    "UTA": "utah-mammoth", "VAN": "vancouver-canucks", "VGK": "vegas-golden-knights",
    "WPG": "winnipeg-jets", "WSH": "washington-capitals",
}

def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())

def section_for_text(text):
    value = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if "injur" in value or "cap hit" in value or "draft pick" in value:
        return None, None
    if "power play" in value or "powerplay" in value or value in {"pp", "pp1", "pp2"}:
        return "PP", 2 if "2" in value else 1
    if "penalty kill" in value or value in {"pk", "pk1", "pk2"}:
        return "PK", 2 if "2" in value else 1
    if "goalie" in value:
        return "G", None
    if "defen" in value or "pairing" in value:
        return "D", None
    if "forward" in value:
        return "F", None
    return None, None

def ordinal_number(value):
    text = str(value or "").lower()
    ordinal = re.search(r"\b([12])(?:st|nd)\b", text)
    match = re.search(r"(?:unit|line|pair|power play|penalty kill)\s*([12])", text)
    if ordinal:
        return int(ordinal.group(1))
    return int(match.group(1)) if match else None

def link_name(node):
    text = " ".join(part.strip() for part in node.xpath(".//text()").getall() if part.strip())
    return re.sub(r"\s+", " ", text).strip()

def parse_dfo_html(html, team, url=None):
    response = TextResponse(url=url or f"https://www.dailyfaceoff.com/teams/{team}", body=html.encode("utf-8"), encoding="utf-8")
    root = response.xpath('//*[@id="line_combos"]') or response
    records = []
    current = None
    section_index = {"F": 0, "D": 0, "G": 0, "PP": 0, "PK": 0}
    order = 0
    nodes = root.xpath(
        ".//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6 | "
        ".//span[@id] | "
        ".//span[contains(translate(string(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'power')] | "
        ".//span[contains(translate(string(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'penalty kill')] | "
        ".//span[contains(translate(string(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'injur')] | "
        ".//a[contains(@href, '/players/')]"
    )
    for node in nodes:
        tag = node.root.tag if getattr(node, "root", None) is not None else ""
        text = link_name(node)
        node_id = node.attrib.get("id", "")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "span"}:
            section_text = f"{text} {node_id}"
            group, unit = section_for_text(section_text)
            if any(value in section_text.lower() for value in ("injur", "cap hit", "draft pick", "cap summary")):
                current = None
                continue
            if group is not None:
                current = (group, unit or ordinal_number(text))
                if group in {"F", "D", "G"}:
                    section_index[group] = 0
            continue
        if current is None or not text:
            continue
        href = node.attrib.get("href", "")
        player_match = re.search(r"/players/(?:[^/]+/)*(\d+)(?:/|$)", href)
        dfo_id = int(player_match.group(1)) if player_match else None
        group, unit = current
        section_index[group] += 1
        index = section_index[group]
        line = None
        line_position = None
        pp_unit = None
        pk_unit = None
        if group == "F":
            line = (index - 1) // 3 + 1
            line_position = ("L", "C", "R")[(index - 1) % 3]
        elif group == "D":
            line = (index - 1) // 2 + 1
        elif group == "PP":
            pp_unit = unit or 1
        elif group == "PK":
            pk_unit = unit or 1
        records.append({
            "team": str(team).upper(), "player_name": text,
            "name_key": normalize_name(text), "dfo_player_id": dfo_id,
            "position_group": group, "line": line, "pp_unit": pp_unit,
            "pk_unit": pk_unit, "line_position": line_position, "dfo_order": order,
        })
        order += 1
    return pd.DataFrame(records, columns=[
        "team", "player_name", "name_key", "dfo_player_id", "position_group",
        "line", "pp_unit", "pk_unit", "line_position", "dfo_order",
    ])

def validate_lineup(frame, team=None, strict=False):
    problems = []
    if frame.empty:
        problems.append("no player links found")
        return problems
    base = frame[frame["position_group"].isin(["F", "D", "G"])]
    if base["name_key"].nunique() != len(base):
        problems.append("duplicate player names")
    counts = base.groupby("position_group")["name_key"].nunique().to_dict()
    for group, expected in (("F", 12), ("D", 6), ("G", 2)):
        if counts.get(group, 0) != expected:
            problems.append(f"{group} count {counts.get(group, 0)}; expected {expected}")
    for group, expected in (("PP", 2), ("PK", 2)):
        units = sorted(frame.loc[frame["position_group"] == group, f"{group.lower()}_unit"].dropna().unique().tolist())
        if strict and units != [1, 2]:
            problems.append(f"{group} units {units}; expected [1, 2]")
    return problems

def fetch_lineup(team, refresh=False, cache_dir=None, timeout=30):
    team = str(team).upper()
    cache = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / "wsba_2026_27_cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"dfo_{team.lower()}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8"), str(path)
    slug = TEAM_SLUGS[team]
    url = f"https://www.dailyfaceoff.com/teams/{slug}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; WSBA lineup scraper/1.0)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text, str(path)

def scrape_lineups(teams=None, refresh=False, cache_dir=None):
    teams = [str(team).upper() for team in (teams or TEAM_SLUGS)]
    frames = []
    errors = []
    for team in teams:
        try:
            html, source = fetch_lineup(team, refresh=refresh, cache_dir=cache_dir)
            frame = parse_dfo_html(html, team, source)
            frame["source"] = source
            frames.append(frame)
            errors.extend({"team": team, "problem": problem} for problem in validate_lineup(frame, team))
        except Exception as exc:
            errors.append({"team": team, "problem": f"fetch/parse: {exc}"})
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, errors
