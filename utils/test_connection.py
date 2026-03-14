import ccxt
import os
from dotenv import load_dotenv
import sys
from pathlib import Path


def test_exchange_connection():
    print("\n=== Testing Exchange Connection ===")

    root = Path(__file__).resolve().parents[1]
    env_candidates = [Path(os.getenv('BOT_ENV_FILE'))] if os.getenv('BOT_ENV_FILE') else []
    env_candidates.append(root / '.env')
    loaded = False
    for p in env_candidates:
        if p.exists():
            load_dotenv(dotenv_path=str(p))
            print(f"Loaded environment from {p}")
            loaded = True
            break
    if not loaded:
        print("WARNING: No env file found. Relying on process environment.")
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: API keys not found in environment file!")
        return False

    try:
        print("Connecting to Binance US...")
        exchange = ccxt.binanceus({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })

        print("Fetching market data...")
        ticker = exchange.fetch_ticker('ETH/USDT')
        print(f"Current ETH price: ${ticker['last']:.2f}")

        print("Fetching account balance...")
        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        eth_balance = balance.get('ETH', {}).get('free', 0)

        print(f"\nAccount Balance:")
        print(f"USDT: ${usdt_balance:.2f}")
        print(f"ETH: {eth_balance:.6f}")

        print("\nConnection test successful!")
        return True

    except Exception as e:
        print(f"\nERROR: Connection test failed!")
        print(f"Error message: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_exchange_connection()
    sys.exit(0 if success else 1)
