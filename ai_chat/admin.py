from django.contrib import admin
from django import forms

from .models import AiConfig, GeminiModel


class AiConfigForm(forms.ModelForm):
    class Meta:
        model = AiConfig
        fields = '__all__'
        widgets = {
            'api_key': forms.PasswordInput(render_value=True),
        }


@admin.register(AiConfig)
class AiConfigAdmin(admin.ModelAdmin):
    form = AiConfigForm
    list_display = ('api_key_preview', 'is_active', 'updated_at')
    list_filter = ('is_active',)
    search_fields = ('api_key',)
    ordering = ('-is_active', '-updated_at')
    fieldsets = (
        ('Google AI Credentials', {
            'fields': ('api_key',),
            'description': 'Enter your Google Gemini API key. Get one free at '
                           '<a href="https://aistudio.google.com/apikey" target="_blank">Google AI Studio</a>.',
        }),
        ('Settings', {
            'fields': ('is_active',),
        }),
    )

    @admin.display(description='API Key')
    def api_key_preview(self, obj):
        if obj.api_key:
            return f'{obj.api_key[:12]}...' if len(obj.api_key) > 12 else obj.api_key
        return '—'


@admin.register(GeminiModel)
class GeminiModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'rank', 'label_short', 'is_active', 'auto_discovered', 'updated_at')
    list_filter = ('is_active', 'auto_discovered')
    search_fields = ('name', 'label')
    ordering = ('rank', 'name')
    list_editable = ('rank', 'is_active')
    fieldsets = (
        ('Model Details', {
            'fields': ('name', 'rank', 'label'),
            'description': 'Models are tried in order of rank (1 = best, tried first). '
                           'The system auto-discovers the latest Google models and adds them here.',
        }),
        ('Status', {
            'fields': ('is_active', 'auto_discovered'),
            'description': 'Uncheck "Active" to skip this model during AI fallback attempts.',
        }),
    )

    @admin.display(description='Label')
    def label_short(self, obj):
        return (obj.label[:60] + '...') if len(obj.label) > 60 else obj.label
