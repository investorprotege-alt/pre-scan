"""
send_briefing.py — Sends the pre-market briefing HTML via Gmail SMTP.

Usage:
  python send_briefing.py --session=asx
  python send_briefing.py --session=us
  python send_briefing.py --session=asx --file=data/briefing_asx_20260416.html

Setup (one-time):
  1. Go to https://myaccount.google.com/apppasswords
  2. Create an App Password named "PreMarket Dashboard"
  3. Paste the 16-char password into config/email_config.json → app_password
  4. Set "enabled": true in config/email_config.json
"""

import argparse
import json
import os
import smtplib
import subprocess
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pytz

# Add scripts dir to path so we can import briefing_gen
sys.path.insert(0, str(Path(__file__).parent))
import briefing_gen

DATA_DIR        = Path(__file__).parent.parent / "data"
CONFIG_PATH     = Path(__file__).parent.parent / "config" / "watchlist.json"
EMAIL_CFG_PATH  = Path(__file__).parent.parent / "config" / "email_config.json"
DATA_DIR.mkdir(exist_ok=True)

AEST = pytz.timezone("Australia/Sydney")


def load_email_config() -> dict:
    if not EMAIL_CFG_PATH.exists():
        return {"enabled": False, "error": "email_config.json not found"}
    with open(EMAIL_CFG_PATH) as f:
        return json.load(f)


def load_watchlist_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def send_via_smtp(subject: str, html_body: str, email_cfg: dict) -> bool:
    """Send HTML email via Gmail SMTP using App Password. Returns True on success."""
    smtp_host  = email_cfg.get("smtp_host",   "smtp.gmail.com")
    smtp_port  = email_cfg.get("smtp_port",   587)
    sender     = email_cfg.get("sender_address",    "beeblegums@gmail.com")
    recipient  = email_cfg.get("recipient_address", "beeblegums@gmail.com")
    password   = email_cfg.get("app_password",  "").replace(" ", "")

    if not password or password == "PASTE_YOUR_APP_PASSWORD_HERE":
        print("ERROR: Gmail App Password not configured.")
        print("  -> Open config/email_config.json and add your App Password.")
        print("  -> Guide: https://myaccount.google.com/apppasswords")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient

    # Plain-text fallback
    plain = f"{subject}\n\nOpen this email in an HTML-capable client to view the full briefing."
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    print(f"  Connecting to {smtp_host}:{smtp_port}...")
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"  Email sent to {recipient}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("ERROR: Gmail authentication failed.")
        print("  -> Check your App Password in config/email_config.json")
        print("  -> Make sure 2-Step Verification is on: https://myaccount.google.com/security")
        print("  -> Create App Password at: https://myaccount.google.com/apppasswords")
        return False
    except smtplib.SMTPException as e:
        print(f"ERROR: SMTP error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error sending email: {e}")
        return False


def open_briefing_in_browser(briefing_path: Path):
    """Open the HTML briefing in the default browser as a fallback."""
    try:
        os.startfile(str(briefing_path))  # Windows
        print(f"  Opened briefing in browser: {briefing_path.name}")
    except Exception:
        try:
            subprocess.run(["start", str(briefing_path)], shell=True, check=True)
        except Exception as e:
            print(f"  Could not open browser: {e}")


def run(session: str, file_path: str | None = None, open_browser: bool = True):
    email_cfg = load_email_config()
    now_aest  = datetime.now(AEST)

    # Generate fresh briefing if no file specified
    if not file_path:
        print(f"Generating {session.upper()} briefing...")
        file_path = briefing_gen.generate_briefing(session)

    briefing_path = Path(file_path)
    if not briefing_path.exists():
        print(f"ERROR: Briefing file not found: {file_path}")
        sys.exit(1)

    with open(briefing_path, encoding="utf-8") as f:
        html_content = f.read()

    session_label = "ASX Pre-Market" if session == "asx" else "US Pre-Market"
    subject = (
        f"{session_label} Briefing - "
        f"{now_aest.strftime('%a %d %b %Y, %H:%M AEST')}"
    )

    print(f"\nBriefing: {briefing_path.name}")
    print(f"Subject:  {subject}")

    # --- Try SMTP send ---
    smtp_enabled = email_cfg.get("enabled", False)
    email_sent   = False

    if smtp_enabled:
        print("\nSending via Gmail SMTP...")
        email_sent = send_via_smtp(subject, html_content, email_cfg)
    else:
        print("\nEmail delivery is disabled.")
        print("  -> To enable: open config/email_config.json, add App Password, set \"enabled\": true")

    # --- Always open in browser too ---
    if open_browser:
        print("\nOpening briefing in browser...")
        open_briefing_in_browser(briefing_path)

    # --- Summary ---
    print("\n" + "="*60)
    print(f"  Session:       {session.upper()}")
    print(f"  Briefing file: {briefing_path}")
    print(f"  Email sent:    {'YES' if email_sent else 'NO (see above)'}")
    print(f"  Browser:       {'Opened' if open_browser else 'Skipped'}")
    print("="*60)

    return {
        "session":       session,
        "briefing_file": str(briefing_path),
        "email_sent":    email_sent,
        "subject":       subject,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session",        choices=["asx", "us"], required=True)
    parser.add_argument("--file",           default=None, help="Pre-generated briefing HTML path")
    parser.add_argument("--no-browser",     action="store_true", help="Don't open in browser")
    args = parser.parse_args()

    result = run(args.session, args.file, open_browser=not args.no_browser)
