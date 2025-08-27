from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os, secrets, base64, httpx, urllib.parse
Y_FANTASY = "https://fantasysports.yahooapis.com/fantasy/v2"
from typing import Optional, List, Tuple
from datetime import datetime

# --- weights (tune any time) ---
INJURY_MULT = {
    "OUT": 0.05, "Doubtful": 0.20, "DOUBTFUL": 0.20, "Questionable": 0.85, "QUESTIONABLE": 0.85,
    "Probable": 0.95, "PROBABLE": 0.95, "IR": 0.05, "PUP": 0.10, "SUSP": 0.50,
}
HOME_MULT = {"home": 1.03, "away": 0.97}

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def need_token(request: Request) -> str:
    token = request.cookies.get("y_at")
    if not token:
        raise HTTPException(status_code=401, detail="Not authorized (no token)")
    return token


app = FastAPI(title="Fantasy Helper Backend (Starter)")
FRONTEND_URL = os.getenv("FRONTEND_PUBLIC_URL", "https://fantasy-helper-frontend.vercel.app")

# Allow all origins for simplicity in the starter (ok on free tiers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # exact origin, not "*"
    allow_credentials=True,        # allow cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE_TOKENS = {}

YAHOO_AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
YAHOO_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"

CLIENT_ID = os.getenv("YAHOO_CLIENT_ID", "YOUR_CLIENT_ID_HERE")
CLIENT_SECRET = os.getenv("YAHOO_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")
REDIRECT_URI = os.getenv("YAHOO_REDIRECT_URI", "http://localhost:8000/auth/yahoo/callback")
FRONTEND_URL = os.getenv("FRONTEND_PUBLIC_URL", "http://localhost:5500")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug/oauth")
def debug_oauth():
    # SAFE to show: just the start of your client id + the redirect the server will send to Yahoo
    return {
        "CLIENT_ID_starts_with": (CLIENT_ID or "")[:10],
        "REDIRECT_URI": REDIRECT_URI
    }


@app.get("/auth/yahoo/login")
def yahoo_login():
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "fspt-r",
        "state": state,
    }
    url = f"{YAHOO_AUTH_URL}?{urllib.parse.urlencode(params)}"
    resp = RedirectResponse(url)
    # keep state in a cookie so a short nap doesn't break the flow
    resp.set_cookie("oauth_state", state, max_age=600, secure=True, httponly=True, samesite="none")
    return resp

@app.get("/auth/yahoo/callback")
async def yahoo_callback(request: Request):
    qs = dict(request.query_params)
    code = qs.get("code")
    state = qs.get("state")
    cookie_state = request.cookies.get("oauth_state")
    if not code or not state or state != cookie_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state or code")

    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "redirect_uri": REDIRECT_URI, "code": code}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(YAHOO_TOKEN_URL, data=data, headers=headers)
        if resp.status_code != 200:
            return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{resp.text}</pre>", status_code=400)
        tokens = resp.json()

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    html = f"""
    <html><body style='font-family:sans-serif;padding:20px;'>
      <h2>Yahoo Authorized 🎉</h2>
      <p>Token received. Go back to the frontend and click "Check Connection".</p>
      <details><summary>Debug token (do not share)</summary><pre>{tokens}</pre></details>
      <p><a href="{FRONTEND_URL}">Return to Frontend</a></p>
    </body></html>
    """
    response = HTMLResponse(html)
    if access_token:
        response.set_cookie("y_at", access_token, max_age=3600, httponly=True, secure=True, samesite="none")
    if refresh_token:
        response.set_cookie("y_rt", refresh_token, max_age=30*24*3600, httponly=True, secure=True, samesite="none")
    return response


@app.get("/auth/check")
def auth_check(request: Request):
    return {"authorized": bool(request.cookies.get("y_at"))}


@app.get("/leagues")
async def leagues(request: Request):
    token = request.cookies.get("y_at")
    if not token:
        raise HTTPException(status_code=401, detail="Not authorized (no token)")

    url = "https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games;game_keys=nfl/leagues?format=json"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Yahoo error {r.status_code}: {r.text[:200]}")

    data = r.json()

    # Try to extract a simple list (league_key + name). If structure changes, we’ll just return raw data.
    leagues_simple = []
    try:
        fc = data.get("fantasy_content", {})
        users = fc.get("users", {})
        u0 = users.get("0", {})
        user = u0.get("user", [])
        games = None
        for item in user:
            if isinstance(item, dict) and "games" in item:
                games = item["games"]
                break
        if games:
            for i in range(int(games.get("count", 0))):
                game = games.get(str(i), {}).get("game", [])
                for entry in game:
                    if isinstance(entry, dict) and "leagues" in entry:
                        leagues = entry["leagues"]
                        for j in range(int(leagues.get("count", 0))):
                            league_list = leagues.get(str(j), {}).get("league", [])
                            li = league_list[0] if league_list and isinstance(league_list, list) else {}
                            leagues_simple.append({"league_key": li.get("league_key"), "name": li.get("name")})
        return {"leagues": leagues_simple or [], "raw": data if not leagues_simple else None}
    except Exception:
        return {"leagues": [], "raw": data}

@app.get("/league/{league_key}/teams")
async def league_teams(league_key: str, request: Request):
    token = need_token(request)
    url = f"{Y_FANTASY}/league/{league_key}/teams?format=json"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Yahoo error {r.status_code}: {r.text[:200]}")
    data = r.json()

    # locate teams section
    fc = data.get("fantasy_content", {})
    league = fc.get("league")
    teams_section = None
    if isinstance(league, list):
        for item in league:
            if isinstance(item, dict) and "teams" in item:
                teams_section = item["teams"]
                break
    elif isinstance(league, dict):
        teams_section = league.get("teams")

    teams_simple = []
    if isinstance(teams_section, dict):
        for k, v in teams_section.items():
            if k == "count" or not isinstance(v, dict):
                continue
            team_val = v.get("team")

            # normalize to a list of dict "chunks"
            chunks = []
            if isinstance(team_val, list):
                # case A: [["dict","dict",...]]
                if team_val and isinstance(team_val[0], list):
                    chunks = team_val[0]
                # case B: ["dict","dict",...]
                elif team_val and isinstance(team_val[0], dict):
                    chunks = team_val

            team_key = name = manager = None
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                if "team_key" in ch and not team_key:
                    team_key = ch["team_key"]
                elif "name" in ch and not name:
                    n = ch["name"]
                    name = n.get("full") if isinstance(n, dict) else n
                elif "managers" in ch and not manager:
                    mgrs = ch["managers"]
                    # managers can be a dict {"0": {"manager": {...}}} OR a list [{"manager": {...}}]
                    if isinstance(mgrs, dict):
                        m = mgrs.get("0", {}).get("manager", {})
                    elif isinstance(mgrs, list) and mgrs and isinstance(mgrs[0], dict):
                        m = mgrs[0].get("manager", {})
                    else:
                        m = {}
                    manager = m.get("nickname") or m.get("guid")

            if team_key or name:
                teams_simple.append({"team_key": team_key, "name": name, "manager": manager})

    return {"teams": teams_simple or [], "raw": None if teams_simple else data}


@app.get("/team/{team_key}/roster")
async def team_roster(team_key: str, request: Request, week: Optional[int] = None):
    token = need_token(request)
    week_part = f";week={week}" if week else ""
    url = f"{Y_FANTASY}/team/{team_key}/roster{week_part}?format=json"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Yahoo error {r.status_code}: {r.text[:200]}")
    data = r.json()

    players = []
    try:
        fc = data.get("fantasy_content", {})
        team_obj = fc.get("team", [])
        roster = team_obj[1].get("roster", {}).get("0", {}).get("players", {}) if len(team_obj) > 1 else {}
        count = int(roster.get("count", 0))
        for i in range(count):
            p = roster.get(str(i), {}).get("player", [])
            meta = p[0] if p else {}
            player_key = meta.get("player_key")
            name = meta.get("name", {}).get("full")
            pos = None
            for entry in p:
                if isinstance(entry, dict) and "primary_position" in entry:
                    pos = entry.get("primary_position")
            players.append({"player_key": player_key, "name": name, "pos": pos})
    except Exception:
        pass
    return {"players": players or [], "raw": data if not players else None}

@app.get("/league/{league_key}/free_agents")
async def league_free_agents(league_key: str, request: Request, count: int = 25, start: int = 0):
    token = need_token(request)
    url = f"{Y_FANTASY}/league/{league_key}/players;status=FA;start={start};count={count}?format=json"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Yahoo error {r.status_code}: {r.text[:200]}")
    data = r.json()

    free_agents = []
    try:
        fc = data.get("fantasy_content", {})
        league_obj = fc.get("league", [])
        players = league_obj[1].get("players", {}) if len(league_obj) > 1 else {}
        cnt = int(players.get("count", 0))
        for i in range(cnt):
            pl = players.get(str(i), {}).get("player", [])
            meta = pl[0] if pl else {}
            player_key = meta.get("player_key")
            name = meta.get("name", {}).get("full")
            pos = None
            for entry in pl:
                if isinstance(entry, dict) and "primary_position" in entry:
                    pos = entry.get("primary_position")
            free_agents.append({"player_key": player_key, "name": name, "pos": pos})
    except Exception:
        pass
    return {"free_agents": free_agents or [], "raw": data if not free_agents else None}
# ===================== Advanced scoring helpers + routes (paste-once) =====================
from typing import Optional, List, Tuple
from datetime import datetime

# small weights you can tweak any time
INJURY_MULT = {
    "OUT": 0.05, "Doubtful": 0.20, "DOUBTFUL": 0.20, "Questionable": 0.85, "QUESTIONABLE": 0.85,
    "Probable": 0.95, "PROBABLE": 0.95, "IR": 0.05, "PUP": 0.10, "SUSP": 0.50,
}
HOME_MULT = {"home": 1.03, "away": 0.97}

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

async def _yahoo_json(url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        return {}
    try:
        return r.json()
    except Exception:
        return {}

def _extract_points_from_yahoo(data: dict) -> Optional[float]:
    # player_points.total
    try:
        p = data.get("fantasy_content", {}).get("player", [])
        for part in p:
            if isinstance(part, dict) and "player_points" in part:
                total = part["player_points"].get("total")
                if total is not None:
                    return float(total)
    except Exception:
        pass
    # occasionally under player_stats.total
    try:
        p = data.get("fantasy_content", {}).get("player", [])
        for part in p:
            if isinstance(part, dict) and "player_stats" in part:
                total = part["player_stats"].get("total")
                if total is not None:
                    return float(total)
    except Exception:
        pass
    return None

async def _get_player_profile(token: str, player_key: str) -> dict:
    """Return {team_abbr, status, injury_status} if Yahoo provides them."""
    url = f"{Y_FANTASY}/player/{player_key}?format=json"
    data = await _yahoo_json(url, token)
    prof = {"team_abbr": None, "status": None, "injury_status": None}
    try:
        p = data.get("fantasy_content", {}).get("player", [])
        meta = p[0] if p and isinstance(p[0], dict) else {}
        prof["team_abbr"] = meta.get("editorial_team_abbr") or meta.get("editorial_team_full_name")
        prof["status"] = meta.get("status")
        for part in p:
            if isinstance(part, dict) and "status_full" in part and not prof["injury_status"]:
                prof["injury_status"] = part.get("status_full")
            if isinstance(part, dict) and "injury_status" in part and not prof["injury_status"]:
                prof["injury_status"] = part.get("injury_status")
    except Exception:
        pass
    return prof

async def _get_week_meta(token: str, player_key: str, week: Optional[int]) -> dict:
    """
    Try to detect opponent + home/away + bye for the given week from Yahoo's player-week stats doc.
    Returns {"opponent": "DAL", "is_home": True/False/None, "is_bye": bool}
    """
    info = {"opponent": None, "is_home": None, "is_bye": False}
    if not week:
        return info
    url = f"{Y_FANTASY}/player/{player_key}/stats;type=week;week={week}?format=json"
    data = await _yahoo_json(url, token)

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = k.lower()
                if "bye" in lk and str(v).lower() in ("1", "true"):
                    info["is_bye"] = True
                if ("opponent" in lk or "opp" in lk):
                    if isinstance(v, str) and len(v) <= 4 and v.isupper():
                        info["opponent"] = info["opponent"] or v
                    if isinstance(v, dict):
                        cand = v.get("abbreviation") or v.get("abbr")
                        if cand and len(cand) <= 4:
                            info["opponent"] = info["opponent"] or cand
                if "is_away" in lk:
                    is_away = (str(v).lower() in ("1", "true"))
                    info["is_home"] = False if is_away else True
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return info

async def _score_player(token: str, player_key: str, pos: str, week: Optional[int], season_fallback: int) -> Tuple[float, float, list]:
    """
    Returns (base_points, final_points, notes[])
    base_points: this week's points/projection if present; else last-season total as proxy.
    final_points: base * context multipliers (injury/bye/home-away/matchup-hook).
    """
    notes = []
    base = 0.0

    # Week baseline
    if week:
        d = await _yahoo_json(f"{Y_FANTASY}/player/{player_key}/stats;type=week;week={week}?format=json", token)
        pts = _extract_points_from_yahoo(d)
        if pts is not None:
            base = float(pts)
            notes.append(f"Week{week}={base:.2f}")

    # Season fallback
    if base == 0.0:
        d2 = await _yahoo_json(f"{Y_FANTASY}/player/{player_key}/stats;type=season;season={season_fallback}?format=json", token)
        pts2 = _extract_points_from_yahoo(d2)
        if pts2 is not None:
            base = float(pts2)
            notes.append(f"Season{season_fallback}={base:.2f}")

    # context multipliers
    prof = await _get_player_profile(token, player_key)
    wk = await _get_week_meta(token, player_key, week)
    mult = 1.0

    inj = prof.get("injury_status") or prof.get("status")
    if inj:
        m = INJURY_MULT.get(inj, 1.0)
        mult *= m
        if m != 1.0:
            notes.append(f"Injury({inj})x{m:.2f}")

    if wk.get("is_bye"):
        mult *= 0.0
        notes.append("Bye x0.00")

    is_home = wk.get("is_home")
    if is_home is True:
        mult *= HOME_MULT["home"]; notes.append(f"Home x{HOME_MULT['home']:.2f}")
    elif is_home is False:
        mult *= HOME_MULT["away"]; notes.append(f"Away x{HOME_MULT['away']:.2f}")

    opp = wk.get("opponent")
    if opp:
        dvp = 1.00  # placeholder; we can wire real DvP / Vegas later
        mult *= dvp
        notes.append(f"Matchup({opp} vs {pos})x{dvp:.2f}")

    final_pts = base * mult
    return base, final_pts, notes

async def _league_slots(token: str, league_key: str) -> tuple[dict, Optional[int]]:
    """Fetch roster slots + current week from league settings."""
    url = f"{Y_FANTASY}/league/{league_key}/settings?format=json"
    data = await _yahoo_json(url, token)
    slots = {}
    current_week = None
    try:
        fc = data.get("fantasy_content", {})
        lg = fc.get("league", [])
        if isinstance(lg, list) and lg and isinstance(lg[0], dict):
            current_week = lg[0].get("current_week")
        settings = None
        for part in lg:
            if isinstance(part, dict) and "settings" in part:
                settings = part["settings"]; break
        if settings:
            rp = settings.get("roster_positions", {})
            for k, v in rp.items():
                if k == "count": continue
                rp_obj = v.get("roster_position", {})
                pos = rp_obj.get("position"); cnt = int(rp_obj.get("count", 0))
                if not pos: continue
                if pos == "W/R/T":
                    slots["FLEX"] = cnt
                elif pos in ("DEF","DEF Team","DST"):
                    slots["DEF"] = slots.get("DEF",0) + cnt
                else:
                    slots[pos] = slots.get(pos,0) + cnt
    except Exception:
        pass
    if not slots:
        slots = {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1,"K":1,"DEF":1}
    return slots, current_week

async def _team_roster_players(token: str, team_key: str, week: Optional[int]) -> List[dict]:
    """Return [{player_key,name,pos}] from Yahoo."""
    wk = f";week={week}" if week else ""
    url = f"{Y_FANTASY}/team/{team_key}/roster{wk}?format=json"
    data = await _yahoo_json(url, token)
    out = []
    try:
        fc = data.get("fantasy_content", {})
        t = fc.get("team", [])
        players = t[1].get("roster", {}).get("0", {}).get("players", {}) if len(t)>1 else {}
        cnt = int(players.get("count", 0))
        for i in range(cnt):
            p = players.get(str(i), {}).get("player", [])
            meta = p[0] if p else {}
            player_key = meta.get("player_key")
            name = meta.get("name", {}).get("full")
            pos = None
            for chunk in p:
                if isinstance(chunk, dict) and "primary_position" in chunk:
                    pos = chunk.get("primary_position")
            out.append({"player_key": player_key, "name": name, "pos": pos})
    except Exception:
        pass
    return out

def _pick_lineup(players: List[dict], slot_counts: dict, flex_positions: List[str], scores: dict):
    by_pos = {}
    for p in players:
        pos = (p.get("pos") or "UTIL").upper()
        by_pos.setdefault(pos, []).append(p)
    for pos, arr in by_pos.items():
        arr.sort(key=lambda x: scores.get(x.get("player_key",""), 0.0), reverse=True)

    starters, used = [], set()
    for pos, cnt in slot_counts.items():
        if pos == "FLEX": continue
        take = by_pos.get(pos, [])
        for i in range(min(cnt, len(take))):
            if take[i]["player_key"] in used: continue
            starters.append({"slot": pos, **take[i], "score": scores.get(take[i]["player_key"], 0.0)})
            used.add(take[i]["player_key"])

    flex_cnt = slot_counts.get("FLEX", 0)
    if flex_cnt > 0:
        pool = []
        for fp in flex_positions:
            for p in by_pos.get(fp, []):
                if p["player_key"] not in used:
                    pool.append(p)
        pool.sort(key=lambda x: scores.get(x["player_key"], 0.0), reverse=True)
        for i in range(min(flex_cnt, len(pool))):
            p = pool[i]
            starters.append({"slot":"FLEX", **p, "score": scores.get(p["player_key"], 0.0)})
            used.add(p["player_key"])

    bench = []
    for arr in by_pos.values():
        for p in arr:
            if p["player_key"] not in used:
                bench.append({**p, "score": scores.get(p["player_key"], 0.0)})
    bench.sort(key=lambda x: x["score"], reverse=True)
    return starters, bench

@app.get("/team/{team_key}/start_sit")
async def start_sit(team_key: str, request: Request, league_key: str, week: Optional[int] = None):
    token = need_token(request)
    slots, current_week = await _league_slots(token, league_key)
    target_week = week or current_week

    players = await _team_roster_players(token, team_key, week=None)
    if not players:
        return {"slots": slots, "week": target_week, "lineup": [], "bench": [], "notes": ["No roster players found."]}

    year = datetime.utcnow().year
    season_fallback = year - 1

    scores, detail = {}, {}
    for p in players:
        pk = p.get("player_key"); pos = (p.get("pos") or "").upper()
        base, final, notes = await _score_player(token, pk, pos, target_week, season_fallback)
        scores[pk] = final
        detail[pk] = {"base": base, "final": final, "notes": notes}

    lineup, bench = _pick_lineup(players, slots, ["RB","WR","TE"], scores)

    for x in lineup:
        d = detail.get(x["player_key"], {})
        x["base"] = d.get("base", 0.0); x["score"] = d.get("final", 0.0); x["explain"] = d.get("notes", [])
    for x in bench:
        d = detail.get(x["player_key"], {})
        x["base"] = d.get("base", 0.0); x["explain"] = d.get("notes", [])

    notes = []
    if target_week is None:
        notes.append("Current week unknown—used season baselines.")
    if all((x.get("score", 0.0) == 0.0) for x in lineup):
        notes.append("No weekly/season points available yet—scores will improve once stats populate.")

    return {"slots": slots, "week": target_week, "lineup": lineup, "bench": bench, "notes": notes}

@app.get("/league/{league_key}/free_agents_ranked")
async def free_agents_ranked(league_key: str, request: Request, count: int = 25, start: int = 0, pos: Optional[str] = None):
    token = need_token(request)
    slots, current_week = await _league_slots(token, league_key)
    target_week = current_week

    # reuse your existing FA route to get the raw list
    base = await league_free_agents(league_key, request, count=count, start=start)
    fa = base.get("free_agents", [])
    if pos:
        fa = [p for p in fa if (p.get("pos") or "").upper() == pos.upper()]

    year = datetime.utcnow().year
    season_fallback = year - 1

    scored = []
    for p in fa:
        pk = p.get("player_key"); ppos = (p.get("pos") or "").upper()
        base_pts, final_pts, notes = await _score_player(token, pk, ppos, target_week, season_fallback)
        scored.append({**p, "base": base_pts, "score": final_pts, "explain": notes})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"week": target_week, "free_agents": scored}
# ===================== end advanced block =====================
# ===================== FA drop recommendations + trade finder =====================

def _lineup_total(starters: list[dict]) -> float:
    return sum(float(x.get("score", 0.0)) for x in starters)

async def _score_players_map(token: str, players: list[dict], week: int | None, season_fallback: int) -> tuple[dict, dict]:
    """Return scores map and detail map for a list of simple {player_key,pos} items."""
    scores, detail = {}, {}
    for p in players:
        pk = p.get("player_key")
        pos = (p.get("pos") or "").upper()
        base, final, notes = await _score_player(token, pk, pos, week, season_fallback)
        scores[pk] = final
        detail[pk] = {"base": base, "final": final, "notes": notes}
    return scores, detail

def _players_without(players: list[dict], drop_key: str) -> list[dict]:
    return [p for p in players if p.get("player_key") != drop_key]

def _players_with(players: list[dict], add_p: dict) -> list[dict]:
    # keep the same shape {player_key,name,pos}
    return players + [{"player_key": add_p.get("player_key"), "name": add_p.get("name"), "pos": add_p.get("pos")}]

@app.get("/league/{league_key}/free_agents_with_drops")
async def free_agents_with_drops(
    league_key: str,
    request: Request,
    team_key: str,
    count: int = 25,
    start: int = 0,
    pos: Optional[str] = None,
    max_drop_candidates: int = 8,
):
    """
    Rank free agents AND, for each FA, recommend the best drop from your roster.
    We simulate 'add FA, drop X', recompute starters, and measure lineup delta.
    """
    token = need_token(request)
    slots, current_week = await _league_slots(token, league_key)
    target_week = current_week
    year = datetime.utcnow().year
    season_fallback = year - 1

    # Your roster + baseline
    my_players = await _team_roster_players(token, team_key, week=None)
    my_scores, _ = await _score_players_map(token, my_players, target_week, season_fallback)
    starters, bench = _pick_lineup(my_players, slots, ["RB","WR","TE"], my_scores)
    baseline_total = _lineup_total(starters)

    # Get FAs (reuse your existing FA route)
    base = await league_free_agents(league_key, request, count=count, start=start)
    fa_list = base.get("free_agents", [])
    if pos:
        fa_list = [p for p in fa_list if (p.get("pos") or "").upper() == pos.upper()]

    # Limit drop candidates: worst bench first, then lowest-scoring starters
    bench_sorted = sorted(
        [{**p, "score": my_scores.get(p.get("player_key",""), 0.0)} for p in bench],
        key=lambda x: x["score"]
    )
    starters_sorted = sorted(
        [{**p, "score": my_scores.get(p.get("player_key",""), 0.0)} for p in starters],
        key=lambda x: x["score"]
    )
    drop_pool = bench_sorted[:max_drop_candidates] + starters_sorted[:max(0, max_drop_candidates - len(bench_sorted))]

    recommendations = []
    for fa in fa_list:
        fa_key = fa.get("player_key")
        fa_pos = (fa.get("pos") or "").upper()
        # score FA
        fa_base, fa_score, fa_notes = await _score_player(token, fa_key, fa_pos, target_week, season_fallback)

        best = None  # (delta, drop_player, new_total, new_starters)
        for d in drop_pool:
            # simulate replace d -> FA
            new_players = _players_with(_players_without(my_players, d["player_key"]), {"player_key": fa_key, "name": fa.get("name"), "pos": fa_pos})
            # re-score: reuse old scores where possible, add FA, remove drop
            scores = dict(my_scores)
            scores.pop(d["player_key"], None)
            scores[fa_key] = fa_score
            # recompute starters for new set
            new_starters, _bench2 = _pick_lineup(new_players, slots, ["RB","WR","TE"], scores)
            new_total = _lineup_total(new_starters)
            delta = new_total - baseline_total
            if (best is None) or (delta > best[0]):
                best = (delta, d, new_total, new_starters)

        rec = {
            "fa": {**fa, "base": fa_base, "score": fa_score, "explain": fa_notes},
            "best_drop": None,
            "delta": 0.0,
            "baseline_total": baseline_total,
            "new_total": baseline_total,
        }
        if best:
            delta, drop_p, new_total, new_lineup = best
            rec["best_drop"] = {"player_key": drop_p["player_key"], "name": drop_p["name"], "pos": drop_p["pos"], "score": drop_p["score"]}
            rec["delta"] = delta
            rec["new_total"] = new_total
            rec["lineup_after"] = new_lineup
        recommendations.append(rec)

    # sort by delta desc, then FA score
    recommendations.sort(key=lambda x: (x.get("delta",0.0), x.get("fa",{}).get("score",0.0)), reverse=True)
    return {"week": target_week, "baseline_total": baseline_total, "recommendations": recommendations}

# ------------------- Trade Finder (1-for-1) -------------------

async def _team_lineup_value(token: str, league_key: str, team_key: str, week: int | None, season_fallback: int):
    slots, _cw = await _league_slots(token, league_key)
    players = await _team_roster_players(token, team_key, week=None)
    scores, _detail = await _score_players_map(token, players, week, season_fallback)
    starters, bench = _pick_lineup(players, slots, ["RB","WR","TE"], scores)
    return _lineup_total(starters), starters, bench, players, scores, slots

def _top_tradeables(bench: list[dict], scores: dict, k: int = 5) -> list[dict]:
    arr = [{**p, "score": scores.get(p.get("player_key",""), 0.0)} for p in bench]
    return sorted(arr, key=lambda x: x["score"], reverse=True)[:k]

@app.get("/league/{league_key}/trade_finder")
async def trade_finder(
    league_key: str,
    request: Request,
    my_team_key: str,
    per_team: int = 5,   # how many bench players from each side to consider
    max_results: int = 20
):
    """
    Suggest 1-for-1 trades that improve BOTH teams' starting lineups.
    We simulate the swap and compute delta for each side.
    """
    token = need_token(request)
    year = datetime.utcnow().year
    season_fallback = year - 1
    # my baseline
    my_total, my_starters, my_bench, my_players, my_scores, my_slots = await _team_lineup_value(token, league_key, my_team_key, week=None, season_fallback=season_fallback)
    my_tradeables = _top_tradeables(my_bench, my_scores, k=per_team)

    # get league teams
    league_teams_resp = await league_teams(league_key, request)
    teams = league_teams_resp.get("teams", [])
    other_teams = [t for t in teams if t.get("team_key") and t["team_key"] != my_team_key]

    suggestions = []
    for T in other_teams:
        their_key = T["team_key"]
        their_total, their_starters, their_bench, their_players, their_scores, their_slots = await _team_lineup_value(token, league_key, their_key, week=None, season_fallback=season_fallback)
        their_tradeables = _top_tradeables(their_bench, their_scores, k=per_team)

        for mine in my_tradeables:
            for theirs in their_tradeables:
                # simulate swap for me
                my_after_players = _players_with(_players_without(my_players, mine["player_key"]), {"player_key": theirs["player_key"], "name": theirs["name"], "pos": theirs["pos"]})
                my_scores2 = dict(my_scores)
                my_scores2.pop(mine["player_key"], None)
                # score incoming for me if missing
                base_in, score_in, _notes_in = await _score_player(token, theirs["player_key"], (theirs.get("pos") or "").upper(), None, season_fallback)
                my_scores2[theirs["player_key"]] = score_in
                my_starters2, _ = _pick_lineup(my_after_players, my_slots, ["RB","WR","TE"], my_scores2)
                my_total2 = _lineup_total(my_starters2)
                my_delta = my_total2 - my_total

                # simulate swap for them
                their_after_players = _players_with(_players_without(their_players, theirs["player_key"]), {"player_key": mine["player_key"], "name": mine["name"], "pos": mine["pos"]})
                their_scores2 = dict(their_scores)
                their_scores2.pop(theirs["player_key"], None)
                base_out, score_out, _notes_out = await _score_player(token, mine["player_key"], (mine.get("pos") or "").upper(), None, season_fallback)
                their_scores2[mine["player_key"]] = score_out
                their_starters2, _ = _pick_lineup(their_after_players, their_slots, ["RB","WR","TE"], their_scores2)
                their_total2 = _lineup_total(their_starters2)
                their_delta = their_total2 - their_total

                if my_delta > 0 and their_delta > 0:
                    suggestions.append({
                        "you_give": {"player_key": mine["player_key"], "name": mine["name"], "pos": mine["pos"], "score": mine["score"]},
                        "you_get":  {"player_key": theirs["player_key"], "name": theirs["name"], "pos": theirs["pos"], "score": theirs["score"]},
                        "your_delta": my_delta,
                        "their_team": T.get("name"),
                        "their_team_key": their_key,
                        "their_delta": their_delta,
                        "fairness": my_delta + their_delta  # quick combined lift
                    })

    suggestions.sort(key=lambda x: x["fairness"], reverse=True)
    return {"suggestions": suggestions[:max_results], "baseline_my_total": my_total}
# ===================== end block =====================

# ===================== Multi-player Trade Finder (2-for-1 / 2-for-2) =====================
from itertools import combinations
from typing import Iterable

def _combos_up_to(items: list[dict], k_max: int) -> Iterable[tuple[dict,...]]:
    for k in range(1, max(1, k_max) + 1):
        for c in combinations(items, k):
            yield c

async def _simulate_side(
    token: str,
    players: list[dict],
    scores: dict[str, float],
    slots: dict,
    give: list[dict],
    get_: list[dict],
    week: int | None,
    season_fallback: int,
    cache: dict[str, float],
):
    """Return (new_total, new_starters, scores2)."""
    # remove outgoing
    new_players = [p for p in players if p.get("player_key") not in {g["player_key"] for g in give}]
    # add incoming (keep same shape)
    for g in get_:
        new_players.append({"player_key": g["player_key"], "name": g["name"], "pos": g.get("pos")})

    # update scores map
    scores2 = dict(scores)
    for g in give:
        scores2.pop(g["player_key"], None)
    for g in get_:
        pk = g["player_key"]
        if pk in scores2:
            continue
        if pk in cache:
            score_in = cache[pk]
        else:
            base_in, score_in, _notes_in = await _score_player(
                token, pk, (g.get("pos") or "").upper(), week, season_fallback
            )
            cache[pk] = score_in
        scores2[pk] = score_in

    starters2, _bench2 = _pick_lineup(new_players, slots, ["RB","WR","TE"], scores2)
    return _lineup_total(starters2), starters2, scores2

@app.get("/league/{league_key}/trade_finder_multi")
async def trade_finder_multi(
    league_key: str,
    request: Request,
    my_team_key: str,
    per_team: int = 6,          # how many candidates to consider from each side
    max_from_me: int = 2,       # up to 2 players sent
    max_from_them: int = 2,     # up to 2 players received
    include_weak_starters: bool = False,  # also allow your weak starters as candidates
    include_their_weak_starters: bool = False,
    max_results: int = 40
):
    """
    Suggest multi-player trades (1-for-1, 2-for-1, 1-for-2, 2-for-2).
    Improves BOTH teams' starting lineups based on your advanced scorer.
    """
    token = need_token(request)
    year = datetime.utcnow().year
    season_fallback = year - 1

    # --- my baseline
    my_total, my_starters, my_bench, my_players, my_scores, my_slots = await _team_lineup_value(
        token, league_key, my_team_key, week=None, season_fallback=season_fallback
    )

    # pick my candidate pool
    my_bench_ranked = sorted(
        [{**p, "score": my_scores.get(p.get("player_key",""), 0.0)} for p in my_bench],
        key=lambda x: x["score"], reverse=True
    )
    my_pool = my_bench_ranked[:per_team]
    if include_weak_starters:
        weak_starters = sorted(
            [{**p, "score": my_scores.get(p.get("player_key",""), 0.0)} for p in my_starters],
            key=lambda x: x["score"]
        )[:max(0, per_team // 2)]
        my_pool = (my_pool + weak_starters)[:per_team]

    # league teams
    league_teams_resp = await league_teams(league_key, request)
    teams = league_teams_resp.get("teams", [])
    other_teams = [t for t in teams if t.get("team_key") and t["team_key"] != my_team_key]

    suggestions = []
    global_cache = {}  # player_key -> score to avoid recomputing

    for T in other_teams:
        their_key = T["team_key"]
        their_total, their_starters, their_bench, their_players, their_scores, their_slots = await _team_lineup_value(
            token, league_key, their_key, week=None, season_fallback=season_fallback
        )
        their_bench_ranked = sorted(
            [{**p, "score": their_scores.get(p.get("player_key",""), 0.0)} for p in their_bench],
            key=lambda x: x["score"], reverse=True
        )
        their_pool = their_bench_ranked[:per_team]
        if include_their_weak_starters:
            their_weak_starters = sorted(
                [{**p, "score": their_scores.get(p.get("player_key",""), 0.0)} for p in their_starters],
                key=lambda x: x["score"]
            )[:max(0, per_team // 2)]
            their_pool = (their_pool + their_weak_starters)[:per_team]

        # try all combos up to the limits
        for give_combo in _combos_up_to(my_pool, max_from_me):
            give_keys = {g["player_key"] for g in give_combo}
            # (avoid giving same player twice if duplicates somehow)
            if len(give_keys) != len(give_combo):
                continue

            for get_combo in _combos_up_to(their_pool, max_from_them):
                get_keys = {g["player_key"] for g in get_combo}
                if len(get_keys) != len(get_combo):
                    continue

                # simulate me
                my_total2, my_starters2, my_scores2 = await _simulate_side(
                    token, my_players, my_scores, my_slots,
                    list(give_combo), list(get_combo),
                    week=None, season_fallback=season_fallback, cache=global_cache
                )
                my_delta = my_total2 - my_total
                if my_delta <= 0:
                    continue

                # simulate them (reverse)
                their_total2, their_starters2, their_scores2 = await _simulate_side(
                    token, their_players, their_scores, their_slots,
                    list(get_combo), list(give_combo),
                    week=None, season_fallback=season_fallback, cache=global_cache
                )
                their_delta = their_total2 - their_total
                if their_delta <= 0:
                    continue

                suggestions.append({
                    "their_team": T.get("name"),
                    "their_team_key": their_key,
                    "you_give": [{"player_key": g["player_key"], "name": g["name"], "pos": g.get("pos"), "score": g.get("score")} for g in give_combo],
                    "you_get":  [{"player_key": g["player_key"], "name": g["name"], "pos": g.get("pos"), "score": g.get("score")} for g in get_combo],
                    "your_delta": my_delta,
                    "their_delta": their_delta,
                    "fairness": my_delta + their_delta,
                    "your_total_after": my_total2,
                    "their_total_after": their_total2,
                })

    # rank by combined lift (fairness)
    suggestions.sort(key=lambda x: x["fairness"], reverse=True)
    return {"baseline_my_total": my_total, "suggestions": suggestions[:max_results]}
# ===================== end multi-player trade finder =====================

