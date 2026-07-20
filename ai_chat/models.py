from django.db import models


# Default seed models (ranked best → worst, updated Jul 2026)
# Priority: best free models with search grounding support first.
# Each entry: (model_name, display_label, rank, supports_grounding)
SEED_MODELS = [
    ("gemini-3.5-flash", "Gemini 3.5 Flash — Best free model with search grounding", 1),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite — Fast with search grounding", 2),
    ("gemini-2.5-flash", "Gemini 2.5 Flash — Reliable grounding fallback", 3),
    ("gemini-2.5-pro", "Gemini 2.5 Pro — Advanced reasoning (may hit free tier limits)", 4),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite — Budget fallback (no grounding)", 5),
]


class GeminiModel(models.Model):
    """Auto-discovered and manually managed Gemini AI models for fallback chaining."""
    name = models.CharField(
        max_length=128,
        unique=True,
        verbose_name='Model Name',
        help_text='e.g. gemini-3.5-flash',
    )
    rank = models.PositiveIntegerField(
        default=999,
        verbose_name='Priority Rank',
        help_text='Lower number = higher priority. 1 = best model, tried first.',
    )
    label = models.CharField(
        max_length=256,
        blank=True,
        default='',
        verbose_name='Display Label',
        help_text='Human-readable description of the model.',
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name='Active in Fallback Chain',
        help_text='Uncheck to skip this model during fallback attempts.',
    )
    auto_discovered = models.BooleanField(
        default=False,
        verbose_name='Auto-Discovered',
        help_text='Was this model automatically found via the Google API?',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Gemini Model'
        verbose_name_plural = 'Gemini Models'
        ordering = ['rank', 'name']

    def __str__(self):
        return f'{self.label or self.name} (rank {self.rank})'

    def save(self, *args, **kwargs):
        if not self.label:
            self.label = self.name
        super().save(*args, **kwargs)


class AiConfig(models.Model):
    """Simplified: just API key + active status. Models are managed separately."""
    api_key = models.CharField(
        max_length=512,
        verbose_name='API Key',
        help_text='Your Google Gemini API key from https://aistudio.google.com/apikey',
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name='Active',
        help_text='Uncheck to disable this configuration',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Google AI Configuration'
        verbose_name_plural = 'Google AI Configurations'

    def __str__(self):
        return f'AI Config ({self.api_key[:12]}...)' if self.api_key else 'AI Config (empty)'
