from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from lstm_model import predict_stock_price
from sentiment_analyzer import get_stock_sentiment

app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# Sample data generator for demo purposes
def generate_sample_data(ticker='DEMO', days=365):
    """Generate sample stock data for demonstration"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 4000 if 'TCS' in ticker.upper() else 150
    
    prices = []
    price = base_price
    for i in range(days):
        change = np.random.randn() * (base_price * 0.02)  # 2% daily volatility
        price = max(price + change, base_price * 0.5)  # Don't go below 50% of base
        prices.append(price)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
        'Low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.randint(1000000, 5000000)) for _ in range(days)]
    }, index=dates)
    
    return df

def calculate_macd(df):
    """Calculate MACD indicators"""
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['signal_line'] = df['MACD'].ewm(span=9).mean()
    return df

def generate_signals(df):
    """Generate buy/sell signals based on MACD crossover"""
    long, short = [], []
    
    for i in range(2, len(df)):
        if df['MACD'].iloc[i] > df['signal_line'].iloc[i] and \
           df['MACD'].iloc[i-1] < df['signal_line'].iloc[i-1]:
            long.append(i)
        elif df['MACD'].iloc[i] < df['signal_line'].iloc[i] and \
             df['MACD'].iloc[i-1] > df['signal_line'].iloc[i-1]:
            short.append(i)
    
    # Get actual entry points (next candle)
    real_long = [i+1 for i in long if i+1 < len(df)]
    real_short = [i+1 for i in short if i+1 < len(df)]
    
    return real_long, real_short

def calculate_profits(df, buy_indices, sell_indices):
    """Calculate profit metrics from buy/sell signals"""
    if not buy_indices or not sell_indices:
        return {
            'total_profit': 0,
            'avg_profit_pct': 0,
            'num_trades': 0,
            'win_rate': 0
        }
    
    buy_prices = df['Open'].iloc[buy_indices]
    sell_prices = df['Open'].iloc[sell_indices]
    
    # Align buy/sell pairs
    if sell_prices.index[0] < buy_prices.index[0]:
        sell_prices = sell_prices.iloc[1:]
    if len(buy_prices) > len(sell_prices):
        buy_prices = buy_prices.iloc[:len(sell_prices)]
    
    profits = [(sell_prices.iloc[i] - buy_prices.iloc[i]) for i in range(len(buy_prices))]
    rel_profits = [profits[i] / buy_prices.iloc[i] * 100 for i in range(len(profits))]
    
    winning_trades = sum(1 for p in profits if p > 0)
    
    return {
        'total_profit': float(sum(profits)),
        'avg_profit_pct': float(np.mean(rel_profits)) if rel_profits else 0,
        'num_trades': len(profits),
        'win_rate': (winning_trades / len(profits) * 100) if profits else 0
    }

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/stock/<ticker>')
def get_stock_data(ticker):
    """Fetch stock data with MACD calculations"""
    try:
        period = request.args.get('period', '1y')
        
        # Map period to days
        period_days = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730
        }
        days = period_days.get(period, 365)
        
        # Generate sample data
        df = generate_sample_data(ticker, days)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 404
        
        # Calculate MACD
        df = calculate_macd(df)
        
        # Prepare response data
        df.index = pd.to_datetime(df.index)
        
        response_data = {
            'dates': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'close': df['Close'].tolist(),
            'macd': df['MACD'].tolist(),
            'signal_line': df['signal_line'].tolist(),
            'open': df['Open'].tolist(),
            'high': df['High'].tolist(),
            'low': df['Low'].tolist()
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/<ticker>')
def get_signals(ticker):
    """Get buy/sell signals for a ticker"""
    try:
        period = request.args.get('period', '1y')
        
        period_days = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730
        }
        days = period_days.get(period, 365)
        
        df = generate_sample_data(ticker, days)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 404
        
        df = calculate_macd(df)
        buy_indices, sell_indices = generate_signals(df)
        
        df.index = pd.to_datetime(df.index)
        
        buy_signals = [
            {
                'date': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(df['Open'].iloc[i]),
                'index': i
            }
            for i in buy_indices
        ]
        
        sell_signals = [
            {
                'date': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(df['Open'].iloc[i]),
                'index': i
            }
            for i in sell_indices
        ]
        
        return jsonify({
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest/<ticker>')
def backtest(ticker):
    """Run backtesting and return profit metrics"""
    try:
        period = request.args.get('period', '1y')
        
        period_days = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730
        }
        days = period_days.get(period, 365)
        
        df = generate_sample_data(ticker, days)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 404
        
        df = calculate_macd(df)
        buy_indices, sell_indices = generate_signals(df)
        
        metrics = calculate_profits(df, buy_indices, sell_indices)
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<ticker>')
def predict_price(ticker):
    """Get LSTM price predictions using actual ML model"""
    try:
        period = request.args.get('period', '1y')
        
        period_days = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730
        }
        days = period_days.get(period, 365)
        
        # Ensure we have enough data for LSTM (need at least 80 days)
        # Add extra days for training
        lstm_days = max(days, 120)  # Minimum 120 days for LSTM
        
        # Generate sample data for LSTM training
        df = generate_sample_data(ticker, lstm_days)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 404
        
        # Use actual LSTM model for predictions (optimized for speed)
        print(f"Training LSTM model for {ticker}... (this may take 10-15 seconds)")
        result = predict_stock_price(df, lookback=30, prediction_days=7, epochs=15)
        
        # Generate future dates if prediction succeeded
        if 'error' not in result:
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=8, freq='D')[1:]
            result['prediction_dates'] = future_dates.strftime('%Y-%m-%d').tolist()
            print(f"LSTM predictions generated successfully for {ticker}")
        else:
            print(f"LSTM prediction failed: {result['error']}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in LSTM prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment/<ticker>')
def get_sentiment(ticker):
    """Get sentiment analysis using actual NewsAPI and TextBlob"""
    try:
        days = int(request.args.get('days', 7))
        
        # Use actual sentiment analysis
        print(f"Fetching news and analyzing sentiment for {ticker}...")
        result = get_stock_sentiment(ticker, days=days)
        
        if 'error' in result:
            print(f"Sentiment analysis note: {result['error']}")
        else:
            print(f"Sentiment analysis complete: {result['sentiment_label']} ({result['overall_sentiment']})")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("MACD Stock Predictor - ML ENHANCED")
    print("=" * 60)
    print("NOTE: Using sample stock data for demonstration")
    print("Yahoo Finance API is not accessible from your network")
    print("=" * 60)
    print("✅ LSTM Price Predictions: ENABLED (Real ML Model)")
    print("✅ Sentiment Analysis: ENABLED (Real NewsAPI)")
    print("=" * 60)
    print("\nServer running at: http://localhost:5000")
    print("Try any ticker symbol - ML features use real models!\n")
    app.run(debug=True, port=5000)
