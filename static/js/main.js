let priceChart = null;
let macdChart = null;
let predictionChart = null;
let sentimentChart = null;

// DOM Elements
const tickerInput = document.getElementById('ticker');
const periodSelect = document.getElementById('period');
const intervalSelect = document.getElementById('interval');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const errorDiv = document.getElementById('error');

// Metric elements
const totalProfitEl = document.getElementById('totalProfit');
const avgProfitEl = document.getElementById('avgProfit');
const numTradesEl = document.getElementById('numTrades');
const winRateEl = document.getElementById('winRate');

// Signal elements
const buySignalsEl = document.getElementById('buySignals');
const sellSignalsEl = document.getElementById('sellSignals');

// Event listeners
analyzeBtn.addEventListener('click', analyzeStock);
tickerInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeStock();
});

async function analyzeStock() {
    const ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) {
        showError('Please enter a stock ticker');
        return;
    }

    showLoading(true);
    hideError();

    try {
        // Fetch stock data, signals, and backtest results in parallel
        const [stockData, signals, backtest, predictions, sentiment] = await Promise.all([
            fetchStockData(ticker),
            fetchSignals(ticker),
            fetchBacktest(ticker),
            fetchPredictions(ticker),
            fetchSentiment(ticker)
        ]);

        // Update charts
        updatePriceChart(stockData, signals);
        updateMACDChart(stockData);
        updatePredictionChart(predictions, stockData);
        updateSentimentDisplay(sentiment);

        // Update metrics
        updateMetrics(backtest);

        // Update signal lists
        updateSignalLists(signals);

    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

async function fetchStockData(ticker) {
    const period = periodSelect.value;
    const interval = intervalSelect.value;

    const response = await fetch(`/api/stock/${ticker}?period=${period}&interval=${interval}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to fetch stock data');
    }
    return response.json();
}

async function fetchSignals(ticker) {
    const period = periodSelect.value;
    const interval = intervalSelect.value;

    const response = await fetch(`/api/signals/${ticker}?period=${period}&interval=${interval}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to fetch signals');
    }
    return response.json();
}

async function fetchBacktest(ticker) {
    const period = periodSelect.value;
    const interval = intervalSelect.value;

    const response = await fetch(`/api/backtest/${ticker}?period=${period}&interval=${interval}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to fetch backtest results');
    }
    return response.json();
}

async function fetchPredictions(ticker) {
    const period = periodSelect.value;
    const interval = intervalSelect.value;

    const response = await fetch(`/api/predict/${ticker}?period=${period}&interval=${interval}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to fetch predictions');
    }
    return response.json();
}

async function fetchSentiment(ticker) {
    const response = await fetch(`/api/sentiment/${ticker}?days=7`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to fetch sentiment');
    }
    return response.json();
}

function updatePriceChart(data, signals) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    // Destroy existing chart
    if (priceChart) {
        priceChart.destroy();
    }

    // Create buy/sell signal datasets
    const buyPoints = signals.buy_signals.map(signal => ({
        x: signal.date,
        y: signal.price
    }));

    const sellPoints = signals.sell_signals.map(signal => ({
        x: signal.date,
        y: signal.price
    }));

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Close Price',
                    data: data.close,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                },
                {
                    label: 'Buy Signals',
                    data: buyPoints,
                    backgroundColor: '#10b981',
                    borderColor: '#10b981',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    showLine: false,
                    pointStyle: 'triangle'
                },
                {
                    label: 'Sell Signals',
                    data: sellPoints,
                    backgroundColor: '#ef4444',
                    borderColor: '#ef4444',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    showLine: false,
                    pointStyle: 'triangle',
                    rotation: 180
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        callback: function (value) {
                            return '₹' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

function updateMACDChart(data) {
    const ctx = document.getElementById('macdChart').getContext('2d');

    // Destroy existing chart
    if (macdChart) {
        macdChart.destroy();
    }

    macdChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'MACD',
                    data: data.macd,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                },
                {
                    label: 'Signal Line',
                    data: data.signal_line,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            }
        }
    });
}

function updatePredictionChart(predictions, stockData) {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    // Destroy existing chart
    if (predictionChart) {
        predictionChart.destroy();
    }

    // Check for errors
    if (predictions.error) {
        document.getElementById('predictionSection').innerHTML = `
            <div class="chart-header">
                <h2>📊 Price Predictions (LSTM)</h2>
            </div>
            <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                ${predictions.error}
            </p>
        `;
        return;
    }

    // Get last 30 days of actual data
    const last30Days = stockData.dates.slice(-30);
    const last30Prices = stockData.close.slice(-30);

    // Combine with predictions
    const allDates = [...last30Days, ...predictions.prediction_dates];
    const actualPrices = [...last30Prices, ...Array(predictions.predictions.length).fill(null)];
    const predictedPrices = [...Array(last30Days.length).fill(null), ...predictions.predictions];
    const upperBound = [...Array(last30Days.length).fill(null), ...predictions.confidence_upper];
    const lowerBound = [...Array(last30Days.length).fill(null), ...predictions.confidence_lower];

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allDates,
            datasets: [
                {
                    label: 'Actual Price',
                    data: actualPrices,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 6
                },
                {
                    label: 'Predicted Price',
                    data: predictedPrices,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Upper Confidence',
                    data: upperBound,
                    borderColor: 'rgba(16, 185, 129, 0.3)',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    borderWidth: 1,
                    fill: '+1',
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Lower Confidence',
                    data: lowerBound,
                    borderColor: 'rgba(16, 185, 129, 0.3)',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#94a3b8',
                        usePointStyle: true,
                        padding: 15
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.3)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        callback: function (value) {
                            return '₹' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

function updateSentimentDisplay(sentiment) {
    const sentimentSection = document.getElementById('sentimentSection');

    // Check for errors
    if (sentiment.error) {
        sentimentSection.innerHTML = `
            <div class="sentiment-header">
                <h2>💭 Market Sentiment</h2>
            </div>
            <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                ${sentiment.error}
            </p>
        `;
        return;
    }

    // Determine sentiment color
    let sentimentColor = '#f59e0b'; // neutral
    if (sentiment.overall_sentiment > 0.1) {
        sentimentColor = '#10b981'; // positive
    } else if (sentiment.overall_sentiment < -0.1) {
        sentimentColor = '#ef4444'; // negative
    }

    // Create sentiment gauge
    const sentimentPercentage = ((sentiment.overall_sentiment + 1) / 2) * 100;

    sentimentSection.innerHTML = `
        <div class="sentiment-header">
            <h2>💭 Market Sentiment</h2>
        </div>
        <div class="sentiment-content">
            <div class="sentiment-gauge">
                <div class="sentiment-score" style="color: ${sentimentColor};">
                    ${sentiment.overall_sentiment.toFixed(3)}
                </div>
                <div class="sentiment-label" style="color: ${sentimentColor};">
                    ${sentiment.sentiment_label}
                </div>
                <div class="sentiment-bar">
                    <div class="sentiment-fill" style="width: ${sentimentPercentage}%; background: ${sentimentColor};"></div>
                </div>
                <div class="sentiment-range">
                    <span>Negative</span>
                    <span>Neutral</span>
                    <span>Positive</span>
                </div>
            </div>
            
            <div class="sentiment-stats">
                <div class="stat-item">
                    <span class="stat-label">Articles Analyzed</span>
                    <span class="stat-value">${sentiment.num_articles}</span>
                </div>
            </div>
            
            ${sentiment.articles && sentiment.articles.length > 0 ? `
                <div class="news-articles">
                    <h3>Recent News</h3>
                    ${sentiment.articles.slice(0, 5).map(article => `
                        <div class="news-item">
                            <div class="news-title">
                                <a href="${article.url}" target="_blank">${article.title}</a>
                            </div>
                            <div class="news-meta">
                                <span class="news-source">${article.source}</span>
                                <span class="news-sentiment" style="color: ${article.sentiment > 0.1 ? '#10b981' : article.sentiment < -0.1 ? '#ef4444' : '#f59e0b'};">
                                    ${article.sentiment > 0 ? '📈' : article.sentiment < 0 ? '📉' : '➡️'} ${article.sentiment.toFixed(2)}
                                </span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

function updateMetrics(backtest) {
    totalProfitEl.textContent = `₹${backtest.total_profit.toFixed(2)}`;
    totalProfitEl.style.color = backtest.total_profit >= 0 ? '#10b981' : '#ef4444';

    avgProfitEl.textContent = `${backtest.avg_profit_pct.toFixed(2)}%`;
    avgProfitEl.style.color = backtest.avg_profit_pct >= 0 ? '#10b981' : '#ef4444';

    numTradesEl.textContent = backtest.num_trades;

    winRateEl.textContent = `${backtest.win_rate.toFixed(2)}%`;
    winRateEl.style.color = backtest.win_rate >= 50 ? '#10b981' : '#ef4444';
}

function updateSignalLists(signals) {
    // Update buy signals
    buySignalsEl.innerHTML = signals.buy_signals.length > 0
        ? signals.buy_signals.map(signal => `
            <div class="signal-item buy">
                <div class="signal-date">${new Date(signal.date).toLocaleString()}</div>
                <div class="signal-price">₹${signal.price.toFixed(2)}</div>
            </div>
        `).join('')
        : '<p style="color: var(--text-secondary);">No buy signals found</p>';

    // Update sell signals
    sellSignalsEl.innerHTML = signals.sell_signals.length > 0
        ? signals.sell_signals.map(signal => `
            <div class="signal-item sell">
                <div class="signal-date">${new Date(signal.date).toLocaleString()}</div>
                <div class="signal-price">₹${signal.price.toFixed(2)}</div>
            </div>
        `).join('')
        : '<p style="color: var(--text-secondary);">No sell signals found</p>';
}

function showLoading(show) {
    loading.classList.toggle('active', show);
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.add('active');
    setTimeout(() => hideError(), 5000);
}

function hideError() {
    errorDiv.classList.remove('active');
}

// Load default stock on page load
window.addEventListener('load', () => {
    analyzeStock();
});
