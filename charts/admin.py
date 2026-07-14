from django.contrib import admin
from django import forms

from .models import BinanceConfig


class BinanceConfigForm(forms.ModelForm):
    class Meta:
        model = BinanceConfig
        fields = '__all__'
        widgets = {
            'api_secret': forms.PasswordInput(render_value=True),
        }


@admin.register(BinanceConfig)
class BinanceConfigAdmin(admin.ModelAdmin):
    form = BinanceConfigForm
    list_display = ('api_key_preview', 'use_testnet', 'is_active', 'updated_at')
    list_filter = ('is_active', 'use_testnet')
    search_fields = ('api_key',)
    ordering = ('-is_active', '-updated_at')
    fieldsets = (
        ('Binance API Credentials', {
            'fields': ('api_key', 'api_secret'),
            'description': 'Enter your Binance API credentials. The secret is masked for security.',
        }),
        ('Settings', {
            'fields': ('use_testnet', 'is_active'),
            'description': 'Configuration options.',
        }),
    )

    @admin.display(description='API Key')
    def api_key_preview(self, obj):
        if obj.api_key:
            return f'{obj.api_key[:12]}...' if len(obj.api_key) > 12 else obj.api_key
        return '—'
