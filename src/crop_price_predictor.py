import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta

class CropPricePredictor:
    def __init__(self, data_path=None):
        """Initialize the CropPricePredictor class"""
        self.data_path = data_path
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 30  # Use 30 days of data to predict next 5 days
        self.prediction_days = 5   # Predict prices for next 5 days
        self.model_path = "crop_price_model"
        self.scaler_path = "price_scaler.pkl"
        
    def load_data(self, data_path=None):
        """Load historical price data for crops"""
        if data_path:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("Data path must be provided")
            
        # Load the data
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df, crop_name=None):
        """Preprocess the data for the specified crop"""
        if crop_name:
            # Filter data for specific crop if provided
            crop_data = df[df['crop_name'] == crop_name]
        else:
            # Use all data if no specific crop is specified
            crop_data = df
            
        # Ensure data is sorted by date
        if 'date' in crop_data.columns:
            crop_data['date'] = pd.to_datetime(crop_data['date'])
            crop_data = crop_data.sort_values('date')
            
        # Extract price column
        prices = crop_data['price'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences for LSTM training
        X, y = self._create_sequences(scaled_prices)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test, crop_data
    
    def _create_sequences(self, data):
        """Create input sequences and target values for LSTM model"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_days + 1):
            # Input sequence
            X.append(data[i:i+self.sequence_length])
            # Target values (next 5 days)
            y.append(data[i+self.sequence_length:i+self.sequence_length+self.prediction_days])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build and compile the LSTM model"""
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        
        # Output layer - predicting next 5 days
        model.add(Dense(units=self.prediction_days))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train the LSTM model on the prepared data"""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
            
        # Reshape y_train to match model output shape
        y_train_reshaped = y_train.reshape(y_train.shape[0], self.prediction_days)
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train_reshaped,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def save_model(self):
        """Save the trained model and scaler"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        self.model.save(os.path.join(self.model_path, "lstm_model.h5"))
        joblib.dump(self.scaler, os.path.join(self.model_path, self.scaler_path))
        
        print(f"Model and scaler saved to {self.model_path}")
    
    def load_model(self):
        """Load a previously trained model and scaler"""
        model_file = os.path.join(self.model_path, "lstm_model.h5")
        scaler_file = os.path.join(self.model_path, self.scaler_path)
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            self.model = tf.keras.models.load_model(model_file)
            self.scaler = joblib.load(scaler_file)
            print("Model and scaler loaded successfully")
            return True
        else:
            print("Model or scaler file not found")
            return False
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            print("Model not trained or loaded")
            return None
            
        # Reshape y_test to match model output shape
        y_test_reshaped = y_test.reshape(y_test.shape[0], self.prediction_days)
        
        # Evaluate the model
        loss = self.model.evaluate(X_test, y_test_reshaped, verbose=0)
        print(f"Test Loss: {loss}")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        predictions_original = self.scaler.inverse_transform(predictions)
        y_test_original = self.scaler.inverse_transform(y_test_reshaped)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(np.square(predictions_original - y_test_original)))
        print(f"Root Mean Squared Error: {rmse}")
        
        return predictions_original, y_test_original, rmse
    
    def predict_next_days(self, recent_data):
        """Predict prices for the next 5 days using the most recent data"""
        if self.model is None:
            print("Model not trained or loaded")
            return None
            
        # Scale the input data
        scaled_data = self.scaler.transform(recent_data.reshape(-1, 1))
        
        # Create input sequence
        X_pred = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        # Make prediction
        prediction = self.model.predict(X_pred)
        
        # Inverse transform to get actual prices
        predicted_prices = self.scaler.inverse_transform(prediction)
        
        return predicted_prices[0]
    
    def visualize_prediction(self, actual_prices, predicted_prices, crop_name=None):
        """Visualize the actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        
        # Plot actual prices
        plt.plot(actual_prices, label='Actual Prices', color='blue')
        
        # Plot predicted prices
        plt.plot([i + len(actual_prices) - len(predicted_prices) for i in range(len(predicted_prices))],
                 predicted_prices, label='Predicted Prices', color='red')
                 
        plt.title(f'Crop Price Prediction for {crop_name if crop_name else "All Crops"}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f"price_prediction_{crop_name if crop_name else 'all_crops'}.png")
        plt.close()
    
    def generate_future_dates(self, last_date):
        """Generate dates for the next 5 days starting from last_date"""
        future_dates = []
        current_date = last_date
        
        for _ in range(self.prediction_days):
            current_date += timedelta(days=1)
            future_dates.append(current_date)
            
        return future_dates
    
    def predict_and_display(self, crop_name=None):
        """Load data, make predictions, and display results"""
        # Load data
        df = self.load_data()
        if df is None:
            return
            
        # Preprocess data
        X_train, X_test, y_train, y_test, crop_data = self.preprocess_data(df, crop_name)
        
        # Check if model exists, if not train a new one
        if not self.load_model():
            print("Training a new model...")
            self.build_model((X_train.shape[1], X_train.shape[2]))
            self.train_model(X_train, y_train)
            self.save_model()
        
        # Evaluate the model
        predicted_prices, actual_prices, rmse = self.evaluate_model(X_test, y_test)
        
        # Get the most recent data for prediction
        recent_data = crop_data['price'].values[-self.sequence_length:]
        
        # Predict next 5 days
        next_5_days_prices = self.predict_next_days(recent_data)
        
        # Get the last date in the dataset
        if 'date' in crop_data.columns:
            last_date = crop_data['date'].iloc[-1]
            future_dates = self.generate_future_dates(last_date)
            
            # Display results
            print("\nPredicted prices for the next 5 days:")
            for i, (date, price) in enumerate(zip(future_dates, next_5_days_prices)):
                print(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")
        else:
            # Display results without dates
            print("\nPredicted prices for the next 5 days:")
            for i, price in enumerate(next_5_days_prices):
                print(f"Day {i+1}: ${price:.2f}")
        
        # Visualize results
        self.visualize_prediction(
            crop_data['price'].values[-30:],  # Last 30 days of actual prices
            next_5_days_prices,
            crop_name
        )
        
        return next_5_days_prices


# Example usage
if __name__ == "__main__":
    print("Crop Price Prediction System")
    print("===========================")
    
    # Initialize the price predictor
    predictor = CropPricePredictor()
    
    # Ask for data path
    data_path = input("Enter the path to historical price data CSV: ")
    predictor.data_path = data_path
    
    # Ask for crop name
    crop_name = input("Enter the crop name to predict (leave blank for all crops): ")
    if crop_name.strip() == "":
        crop_name = None
    
    # Make prediction
    predictor.predict_and_display(crop_name)