
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os, secrets, base64, httpx, urllib.parse

app = FastAPI(title="Fantasy Helper Backend (Starter)")

# Allow all origins for simplicity in the starter (ok on free tiers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    resp.set_cookie("oauth_state", state, max_age=600, secure=True, httponly=True, samesite="lax")
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
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": "authorization_code", "redirect_uri": REDIRECT_URI, "code": code}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(YAHOO_TOKEN_URL, data=data, headers=headers)
        if resp.status_code != 200:
            return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{resp.text}</pre>", status_code=400)
        tokens = resp.json()

    html = f"""
    <html><body style='font-family:sans-serif;padding:20px;'>
      <h2>Yahoo Authorized 🎉</h2>
      <p>Token received. Go back to the frontend and click "Check Connection".</p>
      <details><summary>Debug token (do not share)</summary><pre>{tokens}</pre></details>
      <p><a href="{FRONTEND_URL}">Return to Frontend</a></p>
    </body></html>
    """
    return HTMLResponse(html)

@app.get("/auth/check")
def auth_check():
    return {"authorized": True, "note": "Starter OK. Next: store tokens in a DB."}

@app.get("/leagues")
def leagues():
    # Placeholder demo response
    return {"leagues": [
        {"league_key": "123.l.456789", "name": "Demo League 1"},
        {"league_key": "123.l.654321", "name": "Demo League 2"}
    ]}
