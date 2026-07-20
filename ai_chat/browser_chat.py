"""
Browser AI Chat — Connects to the user's browser via CDP.

Strategy:
  1. CDP: If a browser is ALREADY running with --remote-debugging-port on
     port 9222, opens a NEW TAB in that existing window (best experience).
  2. Auto-Launch: If no browser is on port 9222, automatically launches
     a NEW browser instance with --remote-debugging-port and --temp-profile
     (works with zero setup, auto-closes after getting the response).
  3. If all else fails, returns a helpful error message.

Supports Firefox (primary) and Chromium.
"""

import logging
import time
import json
import re
import socket
import shutil
import subprocess
import tempfile
import os
from urllib.parse import quote
from typing import Optional

logger = logging.getLogger(__name__)

# ── Binary paths ───────────────────────────────────────────────────

FIREFOX_BINARIES = [
    "/usr/bin/firefox", "/snap/bin/firefox", "/usr/local/bin/firefox",
]
CHROMIUM_BINARIES = [
    "/snap/bin/chromium", "/usr/bin/chromium", "/usr/bin/chromium-browser",
]

_CDP_TIMEOUT = 45


def _which(paths: list[str]) -> Optional[str]:
    for p in paths:
        found = shutil.which(p)
        if found:
            return found
        if p.startswith("/") and os.path.exists(p):
            return p
    return None


# ── Port utilities ─────────────────────────────────────────────────

def _is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except (OSError, socket.timeout):
        return False


def _find_free_port(start: int = 9222, end: int = 9240) -> int:
    for port in range(start, end + 1):
        if not _is_port_open(port):
            return port
    return start


# ═════════════════════════════════════════════════════════════════════
#  CDP (Chrome DevTools Protocol) — WebSocket helpers
# ═════════════════════════════════════════════════════════════════════

def _cdp_send(ws, method: str, params: dict = None, msg_id: int = 1) -> None:
    ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))


def _cdp_recv(ws, expect_id: int = None, timeout: float = 15) -> dict:
    from websocket import WebSocketTimeoutException
    ws.settimeout(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = ws.recv()
            msg = json.loads(raw)
            if expect_id is None:
                return msg
            if msg.get("id") == expect_id:
                return msg
            continue
        except WebSocketTimeoutException:
            raise TimeoutError(f"CDP recv timed out for id={expect_id}")
    raise TimeoutError(f"CDP recv timed out for id={expect_id}")


def _cdp_wait_event(ws, event: str, timeout: float = 15) -> dict:
    from websocket import WebSocketTimeoutException
    ws.settimeout(timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = ws.recv()
            msg = json.loads(raw)
            if msg.get("method") == event:
                return msg
        except WebSocketTimeoutException:
            raise TimeoutError(f"CDP event '{event}' timed out")
    raise TimeoutError(f"CDP event '{event}' timed out")


# ═════════════════════════════════════════════════════════════════════
#  BROWSER LAUNCH (auto-launch with --temp-profile)
# ═════════════════════════════════════════════════════════════════════

def _launch_browser(browser_type: str, port: int) -> Optional[subprocess.Popen]:
    """
    Launch a new visible browser with --remote-debugging-port.
    Uses --temp-profile to avoid profile conflicts with running instances.

    Returns the Popen process, or None on failure.
    """
    if browser_type == "firefox":
        binary = _which(FIREFOX_BINARIES)
        if not binary:
            return None
        cmd = [
            binary, "--new-instance", "--temp-profile",
            "--remote-debugging-port", str(port),
            "--remote-allow-origins=*", "about:blank",
        ]

    elif browser_type == "chromium":
        binary = _which(CHROMIUM_BINARIES)
        if not binary:
            return None
        # Chromium doesn't have --temp-profile, use a temp user-data-dir
        tmpdir = tempfile.mkdtemp(prefix="chromium-cdp-")
        cmd = [
            binary,
            f"--remote-debugging-port={port}",
            "--remote-allow-origins=*",
            "--new-window", "--no-first-run",
            f"--user-data-dir={tmpdir}",
            "about:blank",
        ]
    else:
        return None

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(20):  # Wait up to 10 seconds
            time.sleep(0.5)
            if _is_port_open(port):
                logger.info(f"Launched {browser_type} on port {port}")
                return proc
        logger.warning(f"{browser_type} didn't start on port {port}")
        try:
            proc.kill()
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning(f"Failed to launch {browser_type}: {e}")
        return None


def _cleanup_browser(proc: subprocess.Popen):
    """Kill a launched browser process."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════
#  GOOGLE AI MODE — CDP Interaction
# ═════════════════════════════════════════════════════════════════════

_JSEXTRACT = r"""
(() => {
    const c = [];
    for (const s of ['[data-md="181"]','.V3FYCf','.wM6sc','.cHaqb','div[role="region"]','[data-tts="answer"]','[class*="answer"]']) {
        for (const e of document.querySelectorAll(s)) {
            const t = e.textContent.trim();
            if (t.length > 100) c.push({text:t, len:t.length});
        }
    }
    c.sort((a,b)=>b.len-a.len);
    const lines = document.body.innerText.split('\n').filter(l=>l.trim().length>0);
    let best='', cur='', inB=false;
    for (const l of lines) {
        const t = l.trim();
        if (t.length<5) {if(inB&&cur.length>best.length)best=cur; cur=''; inB=false; continue;}
        if (/^(Skip to|Settings|Privacy|Terms|About|Search|Images)/i.test(t)) continue;
        if (!cur&&t.length>20) inB=true;
        if (inB) cur+=t+'\n';
    }
    if (cur.length>best.length) best=cur;
    return JSON.stringify({candidate:c[0]?.text||'', block:best.substring(0,5000)});
})()
"""


def _google_ai_via_cdp(ws, query: str) -> str:
    """Navigate tab to Google AI Mode, poll for AI response text."""
    url = f"https://www.google.com/search?q={quote(query)}&udm=50"

    _cdp_send(ws, "Page.enable", {}, msg_id=1)
    _cdp_recv(ws, timeout=3)
    _cdp_send(ws, "Page.bringToFront", {}, msg_id=2)
    _cdp_recv(ws, expect_id=2, timeout=3)
    _cdp_send(ws, "Page.navigate", {"url": url}, msg_id=3)
    _cdp_recv(ws, expect_id=3, timeout=5)

    try:
        _cdp_wait_event(ws, "Page.frameStoppedLoading", timeout=15)
    except TimeoutError:
        pass
    time.sleep(3)

    deadline = time.time() + _CDP_TIMEOUT
    last = ""
    while time.time() < deadline:
        try:
            _cdp_send(ws, "Runtime.evaluate", {
                "expression": _JSEXTRACT, "returnByValue": True,
            }, msg_id=555)
            resp = _cdp_recv(ws, expect_id=555, timeout=10)
            raw = resp.get("result", {}).get("result", {}).get("value", "{}")
            data = json.loads(raw) if isinstance(raw, str) and len(raw) > 10 else {}
            text = data.get("candidate") or data.get("block") or ""
            if len(text) > len(last):
                last = text
            if len(last) > 200:
                return last
            time.sleep(1.5)
        except Exception:
            time.sleep(2)
    return last


# ═════════════════════════════════════════════════════════════════════
#  RESPONSE CLEANING
# ═════════════════════════════════════════════════════════════════════

def _clean(raw: str) -> str:
    if not raw:
        return ""
    for pat in [
        r"Skip to main content", r"Accessibility help",
        r"AI Mode conversation:", r"AI Mode response is ready",
        r"AI responses may include mistakes",
        r"For financial advice, consult a professional",
        r"Learn more", r"Settings\s+Privacy\s+Terms", r"About\s+Ads\s+Business",
    ]:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE)
    lines, uniq, prev = raw.split("\n"), [], None
    for line in lines:
        s = line.strip()
        if s == prev:
            continue
        uniq.append(s if s else "")
        prev = s
    out, blank = [], False
    for line in uniq:
        if not line:
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(line)
    text = "\n".join(out).strip()
    return text[:8000] + "\n\n[...truncated...]" if len(text) > 8000 else text


# ═════════════════════════════════════════════════════════════════════
#  CDP CONNECTION HELPERS
# ═════════════════════════════════════════════════════════════════════

def _create_tab(debug_url: str) -> tuple:
    """Create a new tab via CDP HTTP endpoint. Returns (tab_id, ws_url)."""
    import requests
    resp = requests.put(f"{debug_url}/json/new", params={"url": "about:blank"}, timeout=5)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to create tab: HTTP {resp.status_code}")
    tab = resp.json()
    return tab.get("id"), tab.get("webSocketDebuggerUrl", "")


def _close_tab(debug_url: str, tab_id: str):
    import requests
    try:
        requests.get(f"{debug_url}/json/close/{tab_id}", timeout=3)
    except Exception:
        pass


def _do_cdp_chat(debug_url: str, browser_name: str, query: str) -> dict:
    """
    Do the full CDP chat flow on a connected browser.
    Creates a tab, navigates to Google AI Mode, extracts response, closes tab.
    """
    from websocket import create_connection
    google_url = f"https://www.google.com/search?q={quote(query)}&udm=50"

    tab_id, ws_url = _create_tab(debug_url)
    if not ws_url:
        return {"success": False}

    try:
        ws = create_connection(ws_url, timeout=10)
        text = _google_ai_via_cdp(ws, query)
        ws.close()
        if text:
            return {
                "success": True,
                "response": _clean(text),
                "browser": browser_name,
                "method": "cdp",
                "google_url": google_url,
                "note": f"Opened a tab in your {browser_name.title()} browser for Google AI Mode.",
            }
    except Exception as e:
        logger.warning(f"CDP chat failed: {e}")
    finally:
        try:
            _close_tab(debug_url, tab_id)
        except Exception:
            pass

    return {"success": False}


# ═════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═════════════════════════════════════════════════════════════════════

def get_google_ai_mode_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote(query)}&udm=50"


def browser_chat(query: str) -> dict:
    """
    Send a query to Google AI Mode using the user's browser.

    Strategy:
      1. CDP: Connect to a browser ALREADY running on port 9222
         (opens new tab in existing window — zero-launch)
      2. Auto-Launch Firefox: Spawn a new Firefox with --temp-profile
         (opens new window — auto-closes after response)
      3. Auto-Launch Chromium: Spawn a new Chromium with temp user-data-dir
         (opens new window — auto-closes after response)

    Returns:
        {"response":"...", "success":true|false, "browser":"firefox|chromium",
         "method":"cdp|launch", "google_url":"...", "note":"..."}
    """
    if not query or not query.strip():
        return {"success": False, "response": "Please enter a question.",
                "browser": None, "method": "none",
                "google_url": None, "note": None}

    query = query.strip()
    google_url = get_google_ai_mode_url(query)
    import requests

    # ════════════════════════════════════════════════════════════════
    #  STRATEGY 1: CDP — existing browser on port 9222
    # ════════════════════════════════════════════════════════════════
    try:
        resp = requests.get(f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=2)
        if resp.status_code == 200:
            product = (resp.json().get("Browser") or "").lower()
            browser_name = "firefox" if ("firefox" in product or "moz" in product) else "chromium"
            result = _do_cdp_chat(f"http://127.0.0.1:{CDP_PORT}", browser_name, query)
            if result.get("success"):
                return result
    except Exception:
        pass

    # ════════════════════════════════════════════════════════════════
    #  STRATEGY 2: Auto-launch Firefox
    # ════════════════════════════════════════════════════════════════
    if _which(FIREFOX_BINARIES):
        port = _find_free_port(9223)
        proc = _launch_browser("firefox", port)
        if proc:
            try:
                time.sleep(0.5)
                result = _do_cdp_chat(f"http://127.0.0.1:{port}", "firefox", query)
                if result.get("success"):
                    return result
            finally:
                _cleanup_browser(proc)

    # ════════════════════════════════════════════════════════════════
    #  STRATEGY 3: Auto-launch Chromium
    # ════════════════════════════════════════════════════════════════
    if _which(CHROMIUM_BINARIES):
        port = _find_free_port(9223)
        proc = _launch_browser("chromium", port)
        if proc:
            try:
                time.sleep(0.5)
                result = _do_cdp_chat(f"http://127.0.0.1:{port}", "chromium", query)
                if result.get("success"):
                    return result
            finally:
                _cleanup_browser(proc)

    # ════════════════════════════════════════════════════════════════
    #  ALL FAILED
    # ════════════════════════════════════════════════════════════════
    return {
        "success": False,
        "response": (
            "⚠️ Could not open Google AI Mode.\n\n"
            "I tried connecting to an existing browser and launching "
            "a new one — both failed.\n\n"
            "Please try:\n"
            "1. Close all Firefox windows\n"
            "2. Run: `firefox --remote-debugging-port 9222`\n"
            "3. Log into the Django app in that Firefox\n"
            "4. Send your message again\n\n"
            "A tab with Google AI Mode is open where you can see the result."
        ),
        "browser": None, "method": "none",
        "google_url": google_url,
        "note": None,
    }
