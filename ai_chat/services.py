"""
Google Gemini AI Chat Service.

Uses the google-genai SDK to communicate with Gemini.
Features:
- Auto-fallback: tries the best active model first, falls back on rate limits
- Auto-discovery: periodically checks Google's API for new free models (async)
- Stale cleanup: removes deprecated models from the active list
- Google Search Grounding: real-time web search like google.com's AI mode
- Model attribution: returns which model generated the response
- No conversation history stored — stateless per request.
"""

import logging
import re
import time
import threading

from django.core.exceptions import SynchronousOnlyOperation
from django.db.utils import OperationalError, ProgrammingError

from .models import AiConfig, GeminiModel, SEED_MODELS
from .google_search import search_ai_mode

logger = logging.getLogger(__name__)

# ── Module-level state for performance ─────────────────────────────────
_seeded = False
_discovery_thread = None
_last_discovery_time = 0
_DISCOVERY_COOLDOWN = 3600

_FALLBACK_MODELS = [
    {"name": name, "label": label, "rank": rank}
    for name, label, rank in SEED_MODELS
]

_CHAT_FAMILIES = ("flash", "pro")


# ═════════════════════════════════════════════════════════════════════
#  CONFIG HELPERS
# ═════════════════════════════════════════════════════════════════════

def get_active_config():
    try:
        return AiConfig.objects.filter(is_active=True).first()
    except (OperationalError, ProgrammingError, SynchronousOnlyOperation):
        return None
    except Exception:
        logger.exception("Unexpected error reading AiConfig")
        return None


def get_api_key():
    config = get_active_config()
    if config and config.api_key:
        return config.api_key.strip()
    return None


# ═════════════════════════════════════════════════════════════════════
#  MODEL DISCOVERY & MANAGEMENT
# ═════════════════════════════════════════════════════════════════════

def _is_chat_model(name: str) -> bool:
    match = re.match(r'^models/gemini-\d+\.\d+-(.+)$', name)
    if not match:
        return False
    family = match.group(1).lower()
    if not any(family.startswith(f) for f in _CHAT_FAMILIES):
        return False
    excludes = ["image", "tts", "robotics", "omni", "computer-use", "preview", "exp"]
    for ex in excludes:
        if ex in family:
            return False
    return True


def seed_default_models():
    global _seeded
    if _seeded:
        return
    try:
        if GeminiModel.objects.count() == 0:
            logger.info("Seeding default Gemini models...")
            for name, label, rank in SEED_MODELS:
                GeminiModel.objects.create(
                    name=name, label=label, rank=rank,
                    is_active=True, auto_discovered=False,
                )
            logger.info(f"Seeded {len(SEED_MODELS)} default models.")
        _seeded = True
    except (OperationalError, ProgrammingError, SynchronousOnlyOperation):
        pass
    except Exception:
        logger.exception("Error seeding default models")


def get_active_models():
    try:
        models = GeminiModel.objects.filter(is_active=True).order_by('rank')
        if models.exists():
            return [
                {"name": m.name, "label": m.label or m.name, "rank": m.rank}
                for m in models
            ]
    except (OperationalError, ProgrammingError, SynchronousOnlyOperation):
        pass
    except Exception:
        logger.exception("Error reading models from DB")
    return list(_FALLBACK_MODELS)


def _fetch_and_update_models(api_key: str):
    global _last_discovery_time
    _last_discovery_time = time.time()
    if not api_key:
        return
    try:
        import requests
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        resp = requests.get(url, params={"key": api_key}, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Model discovery API returned {resp.status_code}")
            return
        data = resp.json()
        api_models = data.get("models", [])
        discovered_names = set()
        for m in api_models:
            name = m.get("name", "")
            supported_methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in supported_methods:
                continue
            if not _is_chat_model(name):
                continue
            model_name = name.replace("models/", "", 1)
            display_name = m.get("displayName", model_name)
            discovered_names.add(model_name)
            if not GeminiModel.objects.filter(name=model_name).exists():
                next_rank = (GeminiModel.objects.order_by("-rank").first().rank
                             if GeminiModel.objects.exists() else len(discovered_names))
                GeminiModel.objects.create(
                    name=model_name, label=display_name,
                    rank=next_rank + 1, is_active=True, auto_discovered=True,
                )
                logger.info(f"Discovered new model: {model_name}")
        known_names = set(GeminiModel.objects.filter(auto_discovered=True).values_list("name", flat=True))
        stale = known_names - discovered_names
        if stale:
            count = GeminiModel.objects.filter(
                name__in=stale, auto_discovered=True, is_active=True
            ).update(is_active=False)
            if count:
                logger.info(f"Deactivated {count} stale model(s): {', '.join(sorted(stale))}")
        logger.info(f"Model discovery complete: {len(discovered_names)} chat models found")
    except ImportError:
        logger.warning("requests library not available")
    except Exception:
        logger.exception("Error during model auto-discovery")


def discover_latest_models(api_key):
    global _last_discovery_time, _discovery_thread
    now = time.time()
    if now - _last_discovery_time < _DISCOVERY_COOLDOWN:
        return
    if _discovery_thread and _discovery_thread.is_alive():
        return
    _discovery_thread = threading.Thread(
        target=_fetch_and_update_models, args=(api_key,), daemon=True,
    )
    _discovery_thread.start()


# ═════════════════════════════════════════════════════════════════════
#  GOOGLE AI MODE SCRAPER (Primary method — free, no API key needed)
#  Routes queries through a headless browser to google.com's AI Mode
#  and returns the AI Overview response.
# ═════════════════════════════════════════════════════════════════════


def _try_google_ai_mode(user_message: str) -> dict:
    """
    Try to get an answer from Google's AI Mode (browser-based).
    Always returns a dict. Check result.get('success') to see if it worked.
    The dict includes 'google_ai_mode_available' flag for the frontend.
    """
    result = search_ai_mode(user_message)
    result["google_ai_mode_available"] = result.get("success", False)
    if result.get("success"):
        logger.info("Got response from Google AI Mode")
    else:
        logger.info(f"Google AI Mode unavailable: {result.get('response', 'unknown')[:80]}")
    return result


# ═════════════════════════════════════════════════════════════════════
#  GOOGLE SEARCH GROUNDING (Secondary — requires API key, uses quota)
# ═════════════════════════════════════════════════════════════════════

def _extract_grounding_sources(response) -> list:
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return []
        grounding_meta = getattr(candidates[0], "grounding_metadata", None)
        if not grounding_meta:
            return []
        chunks = getattr(grounding_meta, "grounding_chunks", None)
        if not chunks:
            return []
        sources = []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                title = getattr(web, "title", None) or ""
                uri = getattr(web, "uri", None) or ""
                if uri:
                    sources.append({"title": title, "url": uri})
        return sources
    except Exception:
        logger.exception("Error extracting grounding sources")
        return []


# ═════════════════════════════════════════════════════════════════════
#  FALLBACK CHAIN
# ═════════════════════════════════════════════════════════════════════

def _get_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def _is_rate_limit_error(error_msg: str) -> bool:
    checks = ["quota", "rate", "resource_exhausted", "429", "too many requests"]
    return any(c in error_msg for c in checks)


def _generate_simple(api_key: str, model_name: str, user_message: str):
    """Call Gemini without grounding (fallback for models that don't support it)."""
    client = _get_client(api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
    )
    if response and response.text:
        return response.text, []
    return None, []


def _generate_with_grounding(api_key: str, model_name: str, user_message: str):
    """
    Call Gemini with Google Search Grounding enabled — like google.com's AI mode.
    Falls back to simple generation if the model doesn't support grounding.
    Returns (response_text, sources_list).
    """
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    google_search_retrieval=types.GoogleSearchRetrieval()
                )]
            ),
        )
        if response and response.text:
            sources = _extract_grounding_sources(response)
            return response.text, sources
        return None, []
    except Exception as e:
        error_msg = str(e).lower()
        if "google_search_retrieval" in error_msg or "is not supported" in error_msg:
            logger.info(f"Grounding not supported for {model_name}, falling back to simple")
            return _generate_simple(api_key, model_name, user_message)
        raise


def get_gemini_response(user_message: str) -> dict:
    """
    Send a message to Gemini and get a response with model attribution.

    PRIORITY ORDER:
    1. Google AI Mode scraper (browser-based, free, no API key needed)
    2. Google Search Grounding via Gemini API (if API key is configured)
    3. Simple Gemini generation (fallback if models don't support grounding)

    Falls back through rate limits automatically.

    Returns:
        {"response": "...", "model": "google-ai-mode",
         "sources": [...], "success": True}
        or
        {"response": "...error...", "model": None, "sources": [], "success": False}
    """

    # ── Fetch API key FIRST, before any browser/Playwright operations ──
    # Playwright's sync API creates an async event loop as a side effect,
    # which causes Django ORM to raise SynchronousOnlyOperation if we
    # try to query the DB afterwards. So we grab the key upfront.
    api_key = get_api_key()

    # ── PRIMARY: Try Google AI Mode (browser-based, free, no API key) ──
    google_result = _try_google_ai_mode(user_message)
    if google_result.get("success"):
        google_result["model"] = "google-ai-mode"
        google_result["model_label"] = "Google AI Mode — Real-time web search"
        google_result["data_source"] = "google_ai_mode"
        google_result["freshness"] = "real_time"
        return google_result

    # ── Build Google AI Mode failure info for the frontend banner ──
    google_ai_mode_info = {
        "available": False,
        "reason": google_result.get("response", "Google AI Mode is currently unavailable."),
        "impact": (
            "Google AI Mode searches the live web for the latest prices, news, and market data. "
            "Without it, the AI relies on its training data which may be weeks or months old."
        ),
    }

    # ── FALLBACK: Try Gemini with Grounding (requires API key) ──
    if not api_key:
        return {
            "response": (
                "⚠️ Google Gemini is not configured and Google AI Mode is unavailable. "
                "Please add your API key in the admin panel at /admin/ai_chat/aiconfig/."
            ),
            "model": None, "sources": [], "success": False,
            "google_ai_mode_info": google_ai_mode_info,
        }

    seed_default_models()
    discover_latest_models(api_key)

    models = get_active_models()
    if not models:
        return {
            "response": "⚠️ No active AI models found. Check /admin/ai_chat/geminimodel/.",
            "model": None, "sources": [], "success": False,
            "google_ai_mode_info": google_ai_mode_info,
        }

    logger.info("Google AI Mode unavailable, falling back to Gemini Grounding...")

    # ── Try each model in rank order with grounding ──────────────
    last_error = None
    last_error_type = None
    attempts = 0

    for model_entry in models:
        model_name = model_entry["name"]
        model_label = model_entry.get("label", model_name)

        try:
            attempts += 1
            response_text, sources = _generate_with_grounding(
                api_key, model_name, user_message
            )

            if response_text:
                logger.info(f"AI response from model: {model_name} (attempt #{attempts})")
                # Determine data freshness: sources present = real-time grounding, empty = knowledge base
                had_grounding = len(sources) > 0
                return {
                    "response": response_text,
                    "model": model_name,
                    "model_label": model_label,
                    "sources": sources,
                    "success": True,
                    "data_source": "google_grounding" if had_grounding else "knowledge_base",
                    "freshness": "real_time" if had_grounding else "knowledge_base",
                    "google_ai_mode_info": google_ai_mode_info,
                }

            last_error = f"Empty response from {model_name}"
            last_error_type = "empty"
            continue

        except ImportError:
            return {"response": "⚠️ AI package not installed.", "model": None, "sources": [], "success": False, "google_ai_mode_info": google_ai_mode_info}

        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Model {model_name} failed (attempt #{attempts}): {error_msg[:120]}")

            if any(x in error_msg for x in ["api_key", "unauthorized", "permission", "key not found"]):
                return {"response": "⚠️ Invalid API key.", "model": None, "sources": [], "success": False, "google_ai_mode_info": google_ai_mode_info}

            if "not found" in error_msg and "model" in error_msg:
                last_error = f"Model '{model_name}' not found"
                last_error_type = "not_found"
                continue

            if "safety" in error_msg or "blocked" in error_msg:
                return {"response": "⚠️ Safety filters blocked the response.", "model": None, "sources": [], "success": False, "google_ai_mode_info": google_ai_mode_info}

            if _is_rate_limit_error(error_msg):
                if "free_tier" in error_msg or "exceeded your current" in error_msg:
                    last_error = f"Daily quota exhausted for {model_name}"
                    last_error_type = "daily_quota"
                else:
                    last_error = f"Rate limit reached for {model_name}"
                    last_error_type = "rate_limit"
                logger.info(f"Rate limit on {model_name}, trying next...")
                continue

            last_error = str(e)[:200]
            last_error_type = "unknown"
            logger.exception(f"Unexpected error with {model_name}")
            continue

    # ── All models exhausted ──
    logger.warning(f"All models exhausted after {attempts} attempts. Last: {last_error_type}")

    if last_error_type == "daily_quota":
        return {
            "response": (
                "⚠️ All Gemini models have exhausted their free tier daily quotas.\n\n"
                "📌 Options:\n"
                "  • Wait for daily quota to reset\n"
                "  • Enable billing at aistudio.google.com to remove limits\n"
                "  • Check /admin/ai_chat/geminimodel/ to see available models"
            ),
            "model": None, "sources": [], "success": False,
            "google_ai_mode_info": google_ai_mode_info,
        }

    if last_error_type == "rate_limit":
        return {
            "response": (
                "⚠️ All models are currently rate-limited. Please wait a moment "
                "and try again, or enable billing at aistudio.google.com."
            ),
            "model": None, "sources": [], "success": False,
            "google_ai_mode_info": google_ai_mode_info,
        }

    return {
        "response": f"🤖 Sorry, I couldn't get a response. Last error: {last_error or 'unknown'}",
        "model": None, "sources": [], "success": False,
        "google_ai_mode_info": google_ai_mode_info,
    }
