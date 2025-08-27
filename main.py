from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os, secrets, base64, httpx, urllib.parse
Y_FANTASY = "https://fantasysports.yahooapis.com/fantasy/v2"

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
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Yahoo error {r.status_code}: {r.text[:200]}")
    data = r.json()

    def get_teams_dict(fc: dict):
        # Handle both shapes:
        #   {"league": [ {meta...}, {"teams": {...}} ]}
        #   {"league": {"teams": {...}}}
        league = fc.get("fantasy_content", {}).get("league")
        if isinstance(league, list):
            for item in league:
                if isinstance(item, dict) and "teams" in item:
                    return item["teams"]
        if isinstance(league, dict):
            return league.get("teams")
        return None

    teams_simple = []
    teams = get_teams_dict(data)
    if isinstance(teams, dict):
        for k, v in teams.items():
            if k == "count":
                continue
            team_list = v.get("team", [])
            team_key = None
            name = None
            manager = None
            for chunk in team_list:
                if not isinstance(chunk, dict):
                    continue
                if "team_key" in chunk and not team_key:
                    team_key = chunk["team_key"]
                if "name" in chunk and not name:
                    n = chunk["name"]
                    name = n.get("full") if isinstance(n, dict) else n
                if "managers" in chunk and not manager:
                    m = chunk["managers"].get("0", {}).get("manager", {})
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


