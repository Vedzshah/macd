"""
Test script to verify yfinance is working
"""
import yfinance as yf
import pandas as pd

print("Testing yfinance connection...")
print("=" * 50)

# Test 1: Simple download
print("\nTest 1: Downloading AAPL data...")
try:
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1mo", interval="1d")
    print(f"✓ Success! Downloaded {len(df)} rows")
    print(df.head())
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Alternative method
print("\nTest 2: Using yf.download...")
try:
    df2 = yf.download("AAPL", period="1mo", interval="1d", progress=False)
    print(f"✓ Success! Downloaded {len(df2)} rows")
    print(df2.head())
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Check internet connection
print("\nTest 3: Checking Yahoo Finance API...")
try:
    import requests
    response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL", timeout=5)
    print(f"✓ API accessible! Status code: {response.status_code}")
except Exception as e:
    print(f"✗ Cannot reach Yahoo Finance: {e}")
    print("This might be a network/firewall issue")
