import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class LSTMPredictor:
    """LSTM-based stock price predictor"""
    
    def __init__(self, lookback=60, prediction_days=7):
        """
        Initialize LSTM predictor
        
        Args:
            lookback: Number of days to look back for prediction
            prediction_days: Number of days to predict into future
        """
        self.lookback = lookback
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train(self, price_data, epochs=50, batch_size=32, verbose=0):
        """
        Train LSTM model on historical price data
        
        Args:
            price_data: DataFrame or array of closing prices
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0=silent, 1=progress bar)
        
        Returns:
            Training history
        """
        # Convert to numpy array if DataFrame
        if isinstance(price_data, pd.DataFrame):
            price_data = price_data['Close'].values
        elif isinstance(price_data, pd.Series):
            price_data = price_data.values
            
        # Reshape and scale data
        price_data = price_data.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(price_data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build and train model
        self.model = self.build_model((X.shape[1], 1))
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                                verbose=verbose, validation_split=0.1)
        
        return history
    
    def predict_future(self, price_data):
        """
        Predict future prices
        
        Args:
            price_data: Recent price data (at least lookback days)
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to numpy array if DataFrame
        if isinstance(price_data, pd.DataFrame):
            price_data = price_data['Close'].values
        elif isinstance(price_data, pd.Series):
            price_data = price_data.values
        
        # Get last lookback days
        recent_data = price_data[-self.lookback:]
        recent_data = recent_data.reshape(-1, 1)
        scaled_data = self.scaler.transform(recent_data)
        
        # Predict future prices
        predictions = []
        current_sequence = scaled_data.copy()
        
        for _ in range(self.prediction_days):
            # Reshape for prediction
            X_pred = current_sequence[-self.lookback:].reshape(1, self.lookback, 1)
            
            # Make prediction
            next_pred = self.model.predict(X_pred, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence, next_pred, axis=0)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # Calculate confidence intervals (simple approach using std)
        last_price = price_data[-1]
        price_std = np.std(price_data[-30:])  # Last 30 days volatility
        
        confidence_upper = predictions + (1.96 * price_std)  # 95% confidence
        confidence_lower = predictions - (1.96 * price_std)
        
        return {
            'predictions': predictions.flatten().tolist(),
            'confidence_upper': confidence_upper.flatten().tolist(),
            'confidence_lower': confidence_lower.flatten().tolist(),
            'last_actual_price': float(last_price)
        }


def predict_stock_price(df, lookback=60, prediction_days=7, epochs=25):
    """
    Convenience function to train and predict in one call
    
    Args:
        df: DataFrame with stock data (must have 'Close' column)
        lookback: Number of days to look back
        prediction_days: Number of days to predict
        epochs: Training epochs (default 25 for speed)
    
    Returns:
        Dictionary with predictions and metadata
    """
    try:
        # Check if we have enough data
        if len(df) < lookback + 50:  # Need enough data for training
            return {
                'error': f'Insufficient data. Need at least {lookback + 50} days, got {len(df)} days.',
                'predictions': [],
                'confidence_upper': [],
                'confidence_lower': []
            }
        
        # Initialize and train model
        predictor = LSTMPredictor(lookback=lookback, prediction_days=prediction_days)
        predictor.train(df['Close'], epochs=epochs, verbose=0)
        
        # Make predictions
        result = predictor.predict_future(df['Close'])
        
        # Add metadata
        result['lookback_days'] = lookback
        result['prediction_days'] = prediction_days
        result['training_samples'] = len(df)
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'predictions': [],
            'confidence_upper': [],
            'confidence_lower': []
        }
