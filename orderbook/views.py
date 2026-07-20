from django.http import JsonResponse

from .services import get_formatted_depth


def depth_json(request):
    """Return order book depth data as JSON for a given symbol."""
    symbol = request.GET.get('symbol', '').strip().upper()
    limit = int(request.GET.get('limit', 20))

    if not symbol:
        return JsonResponse({'error': 'Missing symbol parameter'}, status=400)

    try:
        data = get_formatted_depth(symbol, limit)
        return JsonResponse(data)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
