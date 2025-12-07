from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from lstm_model import predict_stock_price
from sentiment_analyzer import get_stock_sentiment


app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

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
        interval = request.args.get('interval', '1d')
        period = request.args.get('period', '1y')
        
        # Download stock data with retry logic
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}. Please check the ticker symbol.'}), 404
        
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
        interval = request.args.get('interval', '1d')
        period = request.args.get('period', '1y')
        
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}. Please check the ticker symbol.'}), 404
        
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
        interval = request.args.get('interval', '1d')
        period = request.args.get('period', '1y')
        
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}. Please check the ticker symbol.'}), 404
        
        df = calculate_macd(df)
        buy_indices, sell_indices = generate_signals(df)
        
        metrics = calculate_profits(df, buy_indices, sell_indices)
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<ticker>')
def predict_price(ticker):
    """Get LSTM price predictions for a ticker"""
    try:
        interval = request.args.get('interval', '1d')
        period = request.args.get('period', '1y')
        
        # Download stock data
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        
        if df.empty:
            return jsonify({'error': f'No data found for {ticker}. Please check the ticker symbol.'}), 404
        
        # Make predictions
        result = predict_stock_price(df, lookback=60, prediction_days=7, epochs=50)
        
        # Generate future dates
        if 'error' not in result:
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=8, freq='D')[1:]  # Skip first (last_date)
            result['prediction_dates'] = future_dates.strftime('%Y-%m-%d').tolist()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment/<ticker>')
def get_sentiment(ticker):
    """Get sentiment analysis for a ticker"""
    try:
        days = int(request.args.get('days', 7))
        
        # Get sentiment analysis
        result = get_stock_sentiment(ticker, days=days)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

