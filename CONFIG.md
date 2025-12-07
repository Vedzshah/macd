# MACD Stock Predictor - Configuration Guide

## NewsAPI Setup

To enable sentiment analysis, you need a NewsAPI key:

1. Visit https://newsapi.org/
2. Sign up for a free account
3. Get your API key
4. Set it as an environment variable:

### Windows (PowerShell):
```powershell
$env:NEWS_API_KEY="your_api_key_here"
```

### Windows (Command Prompt):
```cmd
set NEWS_API_KEY=your_api_key_here
```

### Linux/Mac:
```bash
export NEWS_API_KEY="your_api_key_here"
```

Alternatively, you can edit `sentiment_analyzer.py` and replace `'YOUR_API_KEY_HERE'` with your actual API key on line 10.

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: TensorFlow installation may take a few minutes.

## Running the Application

```bash
python app.py
```

Then open your browser to: http://localhost:5000

## Features

### 1. MACD Technical Analysis
- Real-time stock data from Yahoo Finance
- MACD indicator calculation
- Buy/Sell signal generation
- Backtesting with profit metrics

### 2. LSTM Price Prediction
- 7-day price forecasting using deep learning
- Confidence intervals for predictions
- Trained on historical data

### 3. Sentiment Analysis
- News article sentiment analysis
- Overall market sentiment score
- Recent news headlines with individual sentiment scores

## Troubleshooting

### LSTM Model Training is Slow
- The first analysis may take 30-60 seconds as the model trains
- Subsequent analyses will be faster
- For production, consider pre-training models

### Sentiment Analysis Not Working
- Verify your NewsAPI key is set correctly
- Free tier has 100 requests/day limit
- Check internet connection

### TensorFlow Warnings
- You may see TensorFlow optimization warnings - these are normal
- The model will still work correctly
