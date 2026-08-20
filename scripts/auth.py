"""
Generate a fresh Fyers access token and write it to .env.

Run this once each morning before downloading data:
    python scripts/auth.py
"""
import sys
import re
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.config import FYERS_APP_ID, FYERS_SECRET_KEY, FYERS_REDIRECT_URI

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _update_env_token(token: str) -> None:
    content = ENV_FILE.read_text(encoding="utf-8")
    if "FYERS_ACCESS_TOKEN=" in content:
        content = re.sub(r"FYERS_ACCESS_TOKEN=.*", f"FYERS_ACCESS_TOKEN={token}", content)
    else:
        content += f"\nFYERS_ACCESS_TOKEN={token}\n"
    ENV_FILE.write_text(content, encoding="utf-8")
    print(f"\n.env updated with new access token.")


def main() -> None:
    if not FYERS_APP_ID or not FYERS_SECRET_KEY or not FYERS_REDIRECT_URI:
        print("ERROR: FYERS_APP_ID, FYERS_SECRET_KEY, and FYERS_REDIRECT_URI must be set in .env")
        sys.exit(1)

    from fyers_apiv3 import fyersModel

    session = fyersModel.SessionModel(
        client_id=FYERS_APP_ID,
        secret_key=FYERS_SECRET_KEY,
        redirect_uri=FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )

    auth_url = session.generate_authcode()
    print("\n--- Fyers Authentication ---")
    print("Opening login URL in your browser...")
    print(f"\n{auth_url}\n")
    print("(If the browser didn't open, copy-paste the URL above manually.)")
    webbrowser.open(auth_url)

    print("\nAfter logging in, Fyers will redirect you to your redirect URI.")
    print("Copy the FULL redirect URL from the browser address bar and paste it here.")
    print("It will look like:  https://youruri/?auth_code=XXXX&state=...")
    print()

    redirect_input = input("Paste the full redirect URL (or just the auth_code): ").strip()

    # Extract auth_code whether user pasted the full URL or just the code
    if redirect_input.startswith("http"):
        parsed = urlparse(redirect_input)
        params = parse_qs(parsed.query)
        auth_code = params.get("auth_code", [None])[0]
        if not auth_code:
            params = parse_qs(parsed.fragment)
            auth_code = params.get("auth_code", [None])[0]
    elif "auth_code=" in redirect_input:
        # User pasted a partial query string like "auth_code=XXX&state=..."
        params = parse_qs(redirect_input)
        auth_code = params.get("auth_code", [None])[0]
    else:
        # Strip trailing &state=... if user pasted the raw token with it appended
        auth_code = redirect_input.split("&")[0].strip()

    if not auth_code:
        print("ERROR: Could not extract auth_code from the input.")
        sys.exit(1)

    print(f"\nExchanging auth code for access token...")
    session.set_token(auth_code)
    resp = session.generate_token()

    if resp.get("code") == 200 or "access_token" in resp:
        token = resp["access_token"]
        print(f"Success! Token: {token[:20]}...")
        _update_env_token(token)
        print("\nYou can now run:  python scripts/download_historical.py")
    else:
        print(f"ERROR: Token exchange failed: {resp}")
        sys.exit(1)


if __name__ == "__main__":
    main()
