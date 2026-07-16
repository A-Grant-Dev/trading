"""
Google AI Mode Scraper.

Uses Playwright to open a headless browser, navigate to Google's
AI Mode (udm=50), and return the AI Overview response.

Anti-detection strategy:
1. playwright-stealth patches (JS-level evasion)
2. Anti-detection init script (additional patches)
3. Persistent profile for cookies (appear as returning user)
4. Firefox fallback if Chromium is blocked
5. User-agent rotation

Thread-safe: fresh browser per request (no shared state).
"""

import logging
import os
import random
import re
import tempfile
import time
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# ── User agents to rotate ───────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
]

# ── Anti-detection JavaScript (injects before any page JS runs) ──
_ANTI_DETECT_SCRIPT = """
// Override webdriver property
Object.defineProperty(navigator, 'webdriver', { get: () => false });
// Override chrome.runtime (headless detection)
window.chrome = { runtime: {} };
// Override permissions
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (params) => (
    params.name === 'notifications' ? Promise.resolve({ state: 'prompt' }) : originalQuery(params)
);
// Override plugins array
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
// Override screen resolution
Object.defineProperty(screen, 'width', { get: () => 1366 });
Object.defineProperty(screen, 'height', { get: () => 768 });
Object.defineProperty(screen, 'availWidth', { get: () => 1366 });
Object.defineProperty(screen, 'availHeight', { get: () => 768 });
// Override webgl vendor/renderer
const getParameter = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(p) {
    if (p === 37445) return 'Intel Inc.';
    if (p === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter(p);
};
// Override webgl2 vendor/renderer (Google checks both)
const getParameter2 = WebGL2RenderingContext.prototype.getParameter;
WebGL2RenderingContext.prototype.getParameter = function(p) {
    if (p === 37445) return 'Intel Inc.';
    if (p === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter2(p);
};
"""

# ── Browser failure cache ─────────────────────────────────────────
_chromium_failed_at = 0.0
_firefox_failed_at = 0.0
_BROWSER_COOLDOWN = 300  # 5 minutes

# ── Persistent profile directory for cookies ──────────────────────
_PROFILE_DIR = os.path.join(tempfile.gettempdir(), "google_ai_mode_profile")


def _ensure_profile_dir():
    """Create the persistent profile directory if it doesn't exist."""
    try:
        os.makedirs(_PROFILE_DIR, exist_ok=True)
    except Exception:
        pass


def _try_chromium(query: str) -> dict:
    """Try to get AI Mode response using Chromium with stealth patches."""
    global _chromium_failed_at

    if _chromium_failed_at and time.time() - _chromium_failed_at < _BROWSER_COOLDOWN:
        return {"response": "", "sources": [], "success": False, "engine": "chromium"}

    from playwright.sync_api import sync_playwright

    pw = browser = context = page = None
    try:
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        ua = random.choice(_USER_AGENTS[:3])  # Only Chrome UAs for Chromium
        _ensure_profile_dir()  # Safe inside try block
        context = browser.new_context(
            user_agent=ua,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="America/New_York",
            storage_state="{}_storage.json".format(_PROFILE_DIR) if os.path.exists("{}_storage.json".format(_PROFILE_DIR)) else None,
            no_viewport=False,
        )

        page = context.new_page()

        # Apply playwright-stealth patches
        try:
            from playwright_stealth import stealth
            stealth(page)
        except Exception:
            pass  # stealth not available, use init script only

        # Apply init script as additional backup
        page.add_init_script(_ANTI_DETECT_SCRIPT)

        search_url = f"https://www.google.com/search?q={quote_plus(query)}&udm=50&gl=us&hl=en"
        logger.info(f"[Chromium] Searching: {query[:60]}...")

        page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)

        # Check for block page
        page_text = page.inner_text("body").lower()
        if "unusual traffic" in page_text or "captcha" in page_text:
            logger.warning("[Chromium] Google blocked the request")
            _chromium_failed_at = time.time()
            return {"response": "", "sources": [], "success": False, "engine": "chromium"}

        # Save storage state for next request (persistent cookies)
        try:
            storage_path = f"{_PROFILE_DIR}_storage.json"
            context.storage_state(path=storage_path)
        except Exception:
            pass

        # Extract the AI response
        response_text = _extract_ai_response(page)
        if response_text:
            sources = _extract_sources(page)
            return {"response": response_text, "sources": sources[:10], "success": True, "engine": "chromium"}

        return {"response": "", "sources": [], "success": False, "engine": "chromium"}

    except Exception as e:
        logger.warning(f"[Chromium] Error: {str(e)[:100]}")
        _chromium_failed_at = time.time()
        return {"response": "", "sources": [], "success": False, "engine": "chromium"}

    finally:
        if page:
            try: page.close()
            except Exception: pass
        if context:
            try: context.close()
            except Exception: pass
        if browser:
            try: browser.close()
            except Exception: pass
        if pw:
            try: pw.stop()
            except Exception: pass


def _try_firefox(query: str) -> dict:
    """Try to get AI Mode response using Firefox (different fingerprint)."""
    global _firefox_failed_at

    if _firefox_failed_at and time.time() - _firefox_failed_at < _BROWSER_COOLDOWN:
        return {"response": "", "sources": [], "success": False, "engine": "firefox"}

    from playwright.sync_api import sync_playwright

    pw = browser = context = page = None
    try:
        pw = sync_playwright().start()
        browser = pw.firefox.launch(
            headless=True,
            args=[],
        )

        ua = random.choice(_USER_AGENTS[3:])  # Firefox UAs
        context = browser.new_context(
            user_agent=ua,
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="America/New_York",
        )

        page = context.new_page()

        # Apply playwright-stealth patches
        try:
            from playwright_stealth import stealth
            stealth(page)
        except Exception:
            pass

        page.add_init_script(_ANTI_DETECT_SCRIPT)

        search_url = f"https://www.google.com/search?q={quote_plus(query)}&udm=50&gl=us&hl=en"
        logger.info(f"[Firefox] Searching: {query[:60]}...")

        page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)

        # Check for block page
        page_text = page.inner_text("body").lower()
        if "unusual traffic" in page_text or "captcha" in page_text:
            logger.warning("[Firefox] Google blocked the request")
            _firefox_failed_at = time.time()
            return {"response": "", "sources": [], "success": False, "engine": "firefox"}

        response_text = _extract_ai_response(page)
        if response_text:
            sources = _extract_sources(page)
            return {"response": response_text, "sources": sources[:10], "success": True, "engine": "firefox"}

        return {"response": "", "sources": [], "success": False, "engine": "firefox"}

    except Exception as e:
        logger.warning(f"[Firefox] Error: {str(e)[:100]}")
        _firefox_failed_at = time.time()
        return {"response": "", "sources": [], "success": False, "engine": "firefox"}

    finally:
        if page:
            try: page.close()
            except Exception: pass
        if context:
            try: context.close()
            except Exception: pass
        if browser:
            try: browser.close()
            except Exception: pass
        if pw:
            try: pw.stop()
            except Exception: pass


def _extract_ai_response(page) -> str | None:
    """Extract the AI Overview response text from the page."""
    try:
        body = page.inner_text("body")
        if not body:
            return None

        lines = [l.strip() for l in body.split("\n")]
        meaningful = [
            l for l in lines
            if len(l) > 15
            and "google" not in l.lower()[:15]
            and not l.startswith(("http://", "https://"))
            and "cookie" not in l.lower()
            and "sign in" not in l.lower()
            and "accept all" not in l.lower()
        ]
        if meaningful:
            combined = " ".join(meaningful[:60])
            combined = re.sub(r'\s+', ' ', combined).strip()
            if len(combined) > 8000:
                combined = combined[:8000] + "..."
            if len(combined) > 50:
                return combined
        return None
    except Exception:
        return None


def _extract_sources(page) -> list:
    """Extract source links from the page."""
    sources = []
    try:
        links = page.locator("a[href^='http']").all()
        seen = set()
        for link in links:
            try:
                url = link.get_attribute("href")
                if url and url not in seen and "google.com" not in url:
                    seen.add(url)
                    title = link.inner_text() or ""
                    if title.strip():
                        sources.append({"title": title.strip()[:150], "url": url})
            except Exception:
                pass
    except Exception:
        pass
    return sources


def search_ai_mode(query: str) -> dict:
    """
    Search Google's AI Mode and return the raw response.

    Strategy order:
    1. Chromium with playwright-stealth + anti-detection init script
    2. Firefox (different browser fingerprint)
    3. Returns failure with useful error message

    Returns:
        {"response": "...", "sources": [...], "success": True}
        or
        {"response": "error message", "sources": [], "success": False}
    """
    # ── Strategy 1: Chromium with stealth ──
    result = _try_chromium(query)
    if result.get("success"):
        logger.info("[AI Mode] Got response via Chromium")
        return result

    # ── Strategy 2: Firefox fallback ──
    result = _try_firefox(query)
    if result.get("success"):
        logger.info("[AI Mode] Got response via Firefox")
        return result

    # ── Both failed — return meaningful message ──
    logger.warning("[AI Mode] All browser strategies failed")
    return {
        "response": "Google AI Mode blocked the automated request. "
                     "This is a Google anti-bot measure, not a bug with this site. "
                     "The AI will fall back to the Gemini API with real-time web search instead.",
        "sources": [],
        "success": False,
    }
