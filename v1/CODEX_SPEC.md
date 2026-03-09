# Crypto Trader - Exchange Integration Spec

## Overview
Add live exchange connectivity to the crypto trading system using CCXT library. Enable paper trading and live trading modes with proper API key management.

## Target Exchange: Coinbase Advanced Trade
**Why Coinbase:**
- US-regulated and available in all 50 states
- Lower fees than basic Coinbase (0.05%-0.40% vs 0.50%-4.50%)
- Good API via CCXT (`coinbase` or `coinbaseexchange`)
- Supports our target pairs: BTC/USD, ETH/USD, SOL/USD
- Paper trading via sandbox mode

**Alternatives:**
- Kraken (`kraken`) - if user prefers
- Gemini (`gemini`) - institutional option

## Tasks

### 1. Exchange Connector Module
Create `src/execution/exchange.py`:

```python
"""
Exchange connector using CCXT.
Handles live/paper mode, API authentication, order execution.
"""

class ExchangeConnector:
    def __init__(self, config: dict):
        """
        Initialize exchange connection.
        
        Args:
            config: Exchange config with:
                - exchange_id: 'coinbase' | 'kraken' | 'gemini'
                - api_key: API key (from env or config)
                - api_secret: API secret
                - mode: 'paper' | 'live'
                - testnet: bool (use sandbox if available)
        """
        pass
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> DataFrame:
        """Fetch OHLCV data from exchange."""
        pass
    
    def get_balance(self) -> dict:
        """Get account balance."""
        pass
    
    def create_order(self, symbol: str, side: str, order_type: str, amount: float, price: float = None) -> dict:
        """Place an order. Returns order object."""
        pass
    
    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an open order."""
        pass
    
    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open orders."""
        pass
    
    def get_order(self, order_id: str, symbol: str) -> dict:
        """Get order status."""
        pass
```

### 2. Update Data Fetcher
Modify `src/data/fetcher.py` to use live exchange data when mode='live':

- Add `use_exchange` parameter
- If `use_exchange=True`, use `ExchangeConnector` instead of yfinance
- Keep synthetic data as fallback

### 3. Update Config
Add to `config.yaml`:

```yaml
exchange:
  id: coinbase  # or kraken, gemini
  mode: paper   # paper | live
  testnet: true # use sandbox
  api_key_env: EXCHANGE_API_KEY  # env var name
  api_secret_env: EXCHANGE_API_SECRET
  rate_limit: true  # respect exchange rate limits
  
execution:
  mode: paper  # keep this for now
  exchange_mode: paper  # separate exchange mode
  dry_run: true  # safety: simulate orders even in paper mode initially
```

### 4. Paper Trading Execution Loop
Create `scripts/run_paper.py`:

```python
"""
Paper trading loop:
1. Fetch latest data from exchange (or use synthetic in paper mode)
2. Run models to generate signals
3. Execute trades via ExchangeConnector (paper mode)
4. Log performance
5. Sleep until next bar
"""
```

### 5. API Key Management
Create `.env.example`:

```
# Exchange API Credentials
# Get these from exchange dashboard
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here

# Optional: Telegram for notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

Add `.env` to `.gitignore`

Use `python-dotenv` to load env vars

### 6. Professional README.md
Create a clean, comprehensive README that covers:

- Project overview (what it does, why it exists)
- Features (models, risk management, paper/live trading)
- Architecture (diagram or clear description)
- Installation (dependencies, setup)
- Configuration (how to set up API keys, customize models)
- Usage (how to run backtests, paper trading, live)
- Models explained (brief description of each)
- Safety & Risk (disclaimers, best practices)
- Roadmap / TODO
- License

**Style:** Professional, minimal, NOT AI-generated looking. Use clear sections, code blocks, avoid excessive emoji or hype language. Think README for a production trading system.

## Testing Checklist

After implementation, verify:

- [ ] Can connect to Coinbase sandbox/paper mode
- [ ] Can fetch OHLCV data via CCXT
- [ ] Can check balance
- [ ] Can place paper orders (market, limit)
- [ ] Can cancel orders
- [ ] Config properly loads API keys from .env
- [ ] Existing models still work with new data source
- [ ] Paper trading loop runs without errors
- [ ] README is clear and professional

## Dependencies to Add

Add to `requirements.txt`:
```
ccxt>=4.2.0
python-dotenv>=1.0.0
```

## Security Notes

- **Never commit API keys** to git
- Use `.env` file for credentials (add to `.gitignore`)
- Start with paper/sandbox mode only
- Add confirmation prompt before switching to live trading
- Implement max position limits even in paper mode

## Deliverables

1. `src/execution/exchange.py` - Exchange connector
2. `scripts/run_paper.py` - Paper trading loop
3. Updated `src/data/fetcher.py` - Use exchange data
4. Updated `config.yaml` - Exchange config
5. `.env.example` - Template for API keys
6. `README.md` - Professional documentation
7. Updated `requirements.txt`

---

**Implementation Priority:**
1. Exchange connector (read-only first: fetch data, balances)
2. Paper trading orders (simulated, logged)
3. README documentation
4. Paper trading loop script

**DO NOT implement live trading yet.** Start with paper/sandbox only.
