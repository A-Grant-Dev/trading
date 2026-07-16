import json
import logging

from django.http import JsonResponse

from .services import get_gemini_response

logger = logging.getLogger(__name__)


def chat(request):
    """
    AJAX endpoint for the AI chat bubble.

    Accepts POST with JSON body: { "message": "user's question" }
    Returns JSON: { "response": "...", "model": "gemini-3.5-flash", "success": true }

    The response includes which model was used so the frontend can display it.
    No conversation history is stored — stateless per request.
    """
    if request.method != "POST":
        return JsonResponse(
            {"response": "Only POST requests are supported.", "success": False},
            status=405,
        )

    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse(
            {"response": "Invalid JSON body. Send { 'message': '...' }", "success": False},
            status=400,
        )

    if not user_message:
        return JsonResponse(
            {"response": "Please enter a message.", "success": False},
        )

    # Limit message length
    if len(user_message) > 4000:
        return JsonResponse(
            {"response": "Message is too long (max 4000 characters).", "success": False},
        )

    result = get_gemini_response(user_message)
    return JsonResponse(result)
