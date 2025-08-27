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


