"""
Google Gemini AI Chat Service.

Uses the google-genai SDK to communicate with Gemini.
Features:
- Multi-model fallback: tries the best active model first, falls back on rate limits
- Google Search Grounding: real-time web search for current data
- Grounding-aware chain: tries models with grounding support first, falls back to simple
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

logger = logging.getLogger(__name__)

# ── Module-level state for performance ─────────────────────────────────
_seeded = False
_discovery_thread = None
_last_discovery_time = 0
_DISCOVERY_COOLDOWN = 3600

# Max models to try in a single request (avoids wasting time on auto-discovered legacy models)
MAX_ATTEMPTS = 5
# If this many consecutive models hit rate limits, abort immediately (same quota)
MAX_CONSECUTIVE_RATE_LIMITS = 3

_FALLBACK_MODELS = [
    {"name": name, "label": label, "rank": rank}
    for name, label, rank, *_ in SEED_MODELS
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
            for name, label, rank, *_ in SEED_MODELS:
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
#  GOOGLE SEARCH GROUNDING
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


def _is_model_not_found(error_msg: str) -> bool:
    """Check if error indicates a model is deprecated/removed/not found."""
    if "not found" in error_msg and "model" in error_msg:
        return True
    if "no longer available" in error_msg or "no longer" in error_msg:
        return True
    if "deprecated" in error_msg:
        return True
    if "not_found" in error_msg or "404" in error_msg:
        return True
    return False


def _generate_simple(api_key: str, model_name: str, user_message: str):
    """Call Gemini without grounding (last resort for models that don't support it)."""
    client = _get_client(api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
    )
    if response and response.text:
        return response.text, []
    return None, []


def _is_model_3x(model_name: str) -> bool:
    """Check if model is Gemini 3.x (uses the new google_search tool format)."""
    return bool(re.match(r'^gemini-3\.', model_name))


def _generate_with_grounding(api_key: str, model_name: str, user_message: str):
    """
    Call Gemini with Google Search Grounding enabled — like google.com's AI mode.

    Uses the appropriate API based on model version:
    - Gemini 3.x: interactions.create() with tools=[{"type": "google_search"}]
    - Gemini 2.x: models.generate_content() with google_search_retrieval tool

    Returns (response_text, sources_list) or (None, []) if grounding fails.
    Does NOT fall back to simple generation — that's handled by the caller.
    """
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=api_key)

        if _is_model_3x(model_name):
            # ── Gemini 3.x: Interactions API with google_search tool ──────────
            # Docs: https://ai.google.dev/gemini-api/docs/google-search
            interaction = client.interactions.create(
                model=model_name,
                input=user_message,
                tools=[{"type": "google_search"}],
            )
            if interaction and getattr(interaction, 'output_text', None):
                sources = []
                # Extract citations from the interaction steps
                steps = getattr(interaction, 'steps', None) or []
                for step in steps:
                    if getattr(step, 'type', None) == 'model_output':
                        content = getattr(step, 'content', None) or []
                        for content_block in content:
                            if getattr(content_block, 'type', None) == 'text':
                                annotations = getattr(content_block, 'annotations', None) or []
                                for annotation in annotations:
                                    if getattr(annotation, 'type', None) == 'url_citation':
                                        url = getattr(annotation, 'url', None)
                                        if url:
                                            sources.append({
                                                "title": getattr(annotation, 'title', '') or '',
                                                "url": url,
                                            })
                return interaction.output_text, sources
            return None, []

        else:
            # ── Gemini 2.x: generate_content with google_search_retrieval ─────
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

        # Catch grounding-not-supported (graceful skip to next model)
        grounding_keywords = [
            "google_search_retrieval", "google_search", "is not supported",
            "not supported", "grounding", "interactions",
        ]
        if any(kw in error_msg for kw in grounding_keywords):
            logger.info(f"Grounding not supported for {model_name}: {error_msg[:150]}")
            return None, []

        # Catch 404 / deprecated / not-found — deactivate and skip gracefully
        if _is_model_not_found(error_msg):
            logger.warning(f"Model not available for {model_name}: {error_msg[:150]}")
            # Deactivate so this model isn't retried on every request
            try:
                updated = GeminiModel.objects.filter(
                    name=model_name, is_active=True
                ).update(is_active=False)
                if updated:
                    logger.info(f"Auto-deactivated unavailable model: {model_name}")
            except Exception:
                pass
            return None, []

        raise


# ═════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def get_gemini_response(user_message: str) -> dict:
    """
    Send a message to Gemini and get a response with model attribution.

    STRATEGY (in order):
    1. Try each model (by rank) with Google Search Grounding enabled
    2. If grounding isn't supported for a model, fall back to simple generation
    3. On rate limits, try the next model in the chain
    4. Last resort: simplest model without grounding

    Returns:
        {"response": "text", "model": "gemini-3.5-flash",
         "model_label": "Gemini 3.5 Flash — ...",
         "sources": [...], "success": True,
         "model_info": {"used": "...", "attempts": [...], "status": "ok"}}
        or error dict with success: False
    """

    api_key = get_api_key()

    if not api_key:
        return {
            "response": (
                "⚠️ Google Gemini is not configured. "
                "Please add your API key in the admin panel at /admin/ai_chat/aiconfig/."
            ),
            "model": None, "sources": [], "success": False,
            "model_info": {
                "status": "no_api_key",
                "message": "No API key configured. Add one in the admin panel.",
            },
        }

    seed_default_models()
    discover_latest_models(api_key)

    models = get_active_models()
    if not models:
        return {
            "response": "⚠️ No active AI models found. Check /admin/ai_chat/geminimodel/.",
            "model": None, "sources": [], "success": False,
            "model_info": {
                "status": "no_models",
                "message": "No active models in database.",
            },
        }

    logger.info(f"Sending message to Gemini with search grounding ({min(len(models), MAX_ATTEMPTS)} models max)...")

    # ── Track which models we try and why they fail ──
    attempts_log = []
    last_error = None
    last_error_type = None
    attempts = 0
    consecutive_rate_limits = 0

    for model_entry in models[:MAX_ATTEMPTS]:
        model_name = model_entry["name"]
        model_label = model_entry.get("label", model_name)

        # Try with grounding first
        try:
            attempts += 1
            response_text, sources = _generate_with_grounding(
                api_key, model_name, user_message
            )

            if response_text:
                had_grounding = len(sources) > 0
                logger.info(
                    f"AI response from {model_name} "
                    f"(attempt #{attempts}, grounding={had_grounding})"
                )

                # Build model info for the frontend notification banner
                model_info = {
                    "used": model_name,
                    "label": model_label,
                    "attempts": attempts_log,
                    "status": "ok",
                    "grounding": had_grounding,
                    "message": None,
                }
                if attempts > 1:
                    # Some models were skipped due to errors
                    failures = [a for a in attempts_log if not a.get("success")]
                    if failures:
                        model_info["message"] = (
                            f"Tried {len(failures)} higher-tier model(s) first: "
                            + "; ".join(
                                f"{f['model']} ({f['reason']})" for f in failures
                            )
                        )

                return {
                    "response": response_text,
                    "model": model_name,
                    "model_label": model_label,
                    "sources": sources,
                    "success": True,
                    "data_source": "google_grounding" if had_grounding else "knowledge_base",
                    "freshness": "real_time" if had_grounding else "knowledge_base",
                    "model_info": model_info,
                }

            last_error = f"Empty response from {model_name}"
            last_error_type = "empty"
            attempts_log.append({
                "model": model_name, "success": False, "reason": "Empty response",
            })
            continue

        except ImportError:
            return {
                "response": "⚠️ AI package not installed. Run: pip install google-genai",
                "model": None, "sources": [], "success": False,
            }

        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"Model {model_name} failed (attempt #{attempts}): {error_msg[:120]}")

            if any(x in error_msg for x in ["api_key", "unauthorized", "permission", "key not found"]):
                return {
                    "response": "⚠️ Invalid API key. Check your key at /admin/ai_chat/aiconfig/.",
                    "model": None, "sources": [], "success": False,
                    "model_info": {"status": "bad_key", "message": "API key is invalid."},
                }

            if _is_model_not_found(error_msg):
                reason = f"Model '{model_name}' is unavailable (deprecated or removed)"
                attempts_log.append({"model": model_name, "success": False, "reason": reason})
                last_error = reason
                last_error_type = "not_found"
                # Auto-deactivate dead models so they aren't retried on next request
                try:
                    updated = GeminiModel.objects.filter(
                        name=model_name, is_active=True
                    ).update(is_active=False)
                    if updated:
                        logger.info(f"Auto-deactivated unavailable model: {model_name}")
                except Exception:
                    pass
                continue

            if "safety" in error_msg or "blocked" in error_msg:
                attempts_log.append({
                    "model": model_name, "success": False,
                    "reason": "Response blocked by safety filters",
                })
                return {
                    "response": "⚠️ Safety filters blocked the response. Try rephrasing.",
                    "model": None, "sources": [], "success": False,
                    "model_info": {
                        "status": "safety_block",
                        "message": f"Blocked on {model_name}",
                        "attempts": attempts_log,
                    },
                }

            if _is_rate_limit_error(error_msg):
                consecutive_rate_limits += 1
                if "free_tier" in error_msg or "exceeded your current" in error_msg:
                    reason = "Daily quota exhausted"
                else:
                    reason = "Rate limit reached"
                attempts_log.append({"model": model_name, "success": False, "reason": reason})
                last_error = f"{reason} for {model_name}"
                last_error_type = "rate_limit"
                logger.info(f"{reason} on {model_name} ({consecutive_rate_limits}/{MAX_CONSECUTIVE_RATE_LIMITS}), trying next...")

                # Early abort: if all models share the same quota, no point continuing
                if consecutive_rate_limits >= MAX_CONSECUTIVE_RATE_LIMITS:
                    logger.warning(f"{consecutive_rate_limits} consecutive rate limits. Aborting model chain.")
                    break

                continue
            else:
                consecutive_rate_limits = 0  # Reset on any non-rate-limit error

            reason = str(e)[:200]
            attempts_log.append({"model": model_name, "success": False, "reason": reason})
            last_error = reason
            last_error_type = "unknown"
            logger.exception(f"Unexpected error with {model_name}")
            continue

    # ── All models failed grounding: try simple generation as last resort ──
    first_model = models[0]["name"] if models else None
    first_model_label = models[0].get("label", first_model) if models else None

    logger.warning(
        f"All {attempts} models failed grounding. Trying simple generation with {first_model}..."
    )

    if first_model:
        try:
            response_text, _ = _generate_simple(api_key, first_model, user_message)
            if response_text:
                logger.info(f"Last-resort response from {first_model} (no grounding)")
                model_info = {
                    "used": first_model,
                    "label": first_model_label,
                    "attempts": attempts_log,
                    "status": "ok",
                    "grounding": False,
                    "message": (
                        f"None of the {attempts} model(s) support search grounding. "
                        f"Using {first_model} without live web search."
                    ),
                }
                return {
                    "response": response_text,
                    "model": first_model,
                    "model_label": first_model_label,
                    "sources": [],
                    "success": True,
                    "data_source": "knowledge_base",
                    "freshness": "knowledge_base",
                    "model_info": model_info,
                }
        except Exception as e:
            last_error = str(e)[:200]
            attempts_log.append({
                "model": first_model, "success": False,
                "reason": f"Simple generation also failed: {last_error}",
            })

    # ── Everything failed ──
    logger.warning(f"All models exhausted after {attempts} attempts. Last: {last_error_type}")

    model_info = {
        "status": "all_exhausted",
        "attempts": attempts_log,
        "message": None,
    }

    if last_error_type == "rate_limit":
        model_info["message"] = (
            "All models are rate-limited. Wait a moment or enable billing "
            "at aistudio.google.com."
        )
        return {
            "response": (
                "⚠️ All Gemini models are currently rate-limited.\n\n"
                "📌 Options:\n"
                "  • Wait a moment and try again\n"
                "  • Enable billing at aistudio.google.com to remove free tier limits\n"
                "  • Check /admin/ai_chat/geminimodel/ to see available models"
            ),
            "model": None, "sources": [], "success": False,
            "model_info": model_info,
        }

    if last_error_type == "not_found":
        model_info["message"] = (
            "The configured models may be outdated. Run model auto-discovery "
            "or update them in the admin panel."
        )

    return {
        "response": f"🤖 Sorry, I couldn't get a response. Last error: {last_error or 'unknown'}",
        "model": None, "sources": [], "success": False,
        "model_info": model_info,
    }
