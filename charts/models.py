from django.db import models


class BinanceConfig(models.Model):
    api_key = models.CharField(
        max_length=255,
        verbose_name='API Key',
        help_text='Your Binance API key',
    )
    api_secret = models.CharField(
        max_length=512,
        verbose_name='API Secret',
        help_text='Your Binance API secret key',
    )
    use_testnet = models.BooleanField(
        default=False,
        verbose_name='Use Testnet',
        help_text='Enable to use Binance testnet (for development/testing)',
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name='Active',
        help_text='Uncheck to disable this configuration',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Binance API Configuration'
        verbose_name_plural = 'Binance API Configurations'

    def __str__(self):
        return f'Binance Config ({self.api_key[:8]}...)' if self.api_key else 'Binance Config (empty)'
