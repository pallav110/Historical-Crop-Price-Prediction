import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime, timedelta

class CropPricePredictor:
    def __init__(self, data_path=None):
        """Initialize the CropPricePredictor class"""
        self.data_path = data_path
        self.model = None
        # Using StandardScaler instead of MinMaxScaler for better handling of outliers
        self.scaler = StandardScaler()
        self.sequence_length = 60  # Increased from 30 to 60 for better context
        self.prediction_days = 5   # Predict prices for next 5 days
        self.model_path = "crop_price_model"
        self.scaler_path = "price_scaler.pkl"
        
    def load_data(self, data_path=None):
        """Load historical price data for crops"""
        if data_path:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("Data path must be provided")
            
        try:
            df = pd.read_csv(self.data_path)
            # Check if data has enough rows
            if len(df) < self.sequence_length + self.prediction_days:
                raise ValueError(f"Data contains only {len(df)} rows, but at least {self.sequence_length + self.prediction_days} are required")
            
            # Check for missing values
            if df.isnull().sum().sum() > 0:
                print(f"Warning: Found {df.isnull().sum().sum()} missing values. Filling with forward fill method.")
                df = df.ffill().bfill()  # Forward fill then backward fill any remaining NaNs
                
            # Check for price column
            if 'price' not in df.columns:
                raise ValueError("Data must contain a 'price' column")
                
            # Add basic validation for price data
            if df['price'].min() < 0:
                print(f"Warning: Negative prices found in data. Min price: {df['price'].min()}")
                
            if df['price'].max() > 1e6:
                print(f"Warning: Very high prices found in data. Max price: {df['price'].max()}")
                
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df, crop_name=None):
        """Preprocess the data for the specified crop"""
        if crop_name:
            # Filter data for specific crop if provided
            crop_data = df[df['crop_name'] == crop_name]
            if len(crop_data) == 0:
                available_crops = df['crop_name'].unique()
                raise ValueError(f"No data found for crop '{crop_name}'. Available crops: {available_crops}")
        else:
            # Use all data if no specific crop is specified
            crop_data = df
            
        # Add additional features
        if 'date' in crop_data.columns:
            crop_data['date'] = pd.to_datetime(crop_data['date'])
            crop_data = crop_data.sort_values('date')
            
            # Extract date features
            crop_data['month'] = crop_data['date'].dt.month
            crop_data['day_of_week'] = crop_data['date'].dt.dayofweek
            crop_data['day_of_year'] = crop_data['date'].dt.dayofyear
            
        # Calculate rolling statistics for price
        crop_data['price_7d_mean'] = crop_data['price'].rolling(window=7, min_periods=1).mean()
        crop_data['price_30d_mean'] = crop_data['price'].rolling(window=30, min_periods=1).mean()
        crop_data['price_7d_std'] = crop_data['price'].rolling(window=7, min_periods=1).std().fillna(0)
        
        # Extract features
        feature_columns = ['price', 'price_7d_mean', 'price_30d_mean', 'price_7d_std']
        
        if 'month' in crop_data.columns:
            feature_columns.extend(['month', 'day_of_week', 'day_of_year'])
            
        # Create feature matrix
        features = crop_data[feature_columns].values
        
        # Scale the data
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences for LSTM training
        X, y = self._create_sequences(scaled_features)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Store the original prices for reference
        original_prices = crop_data['price'].values
        
        return X_train, X_test, y_train, y_test, crop_data, original_prices
    
    def _create_sequences(self, data):
        """Create input sequences and target values for LSTM model"""
        X, y = [], []
        price_index = 0  # Assuming price is the first column
        
        for i in range(len(data) - self.sequence_length - self.prediction_days + 1):
            # Input sequence (all features)
            X.append(data[i:i+self.sequence_length])
            # Target values (next 5 days of prices only)
            y_vals = data[i+self.sequence_length:i+self.sequence_length+self.prediction_days, price_index:price_index+1]
            y.append(y_vals)
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build and compile the LSTM model with improvements"""
        model = Sequential()
        
        # First LSTM layer with more units and proper return sequences
        model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=64, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Third LSTM layer
        model.add(LSTM(units=32))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Dense layers for better feature extraction
        model.add(Dense(units=32, activation='relu'))
        model.add(BatchNormalization())
        
        # Output layer - predicting next 5 days
        model.add(Dense(units=self.prediction_days))
        
        # Use a more robust optimizer with a smaller learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Compile the model with Huber loss instead of MSE for robustness to outliers
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the LSTM model with callbacks for better training"""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
            
        # Reshape y_train to match model output shape
        y_train_reshaped = y_train.reshape(y_train.shape[0], self.prediction_days)
        y_test_reshaped = y_test.reshape(y_test.shape[0], self.prediction_days)
        
        # Create callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint to save the best model
            ModelCheckpoint(
                filepath=os.path.join(self.model_path, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Reduce learning rate when a metric has stopped improving
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train_reshaped,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test_reshaped),
            callbacks=callbacks,
            verbose=1
        )
        
        # Load the best model
        self.model = tf.keras.models.load_model(os.path.join(self.model_path, 'best_model.h5'))
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot the training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_path, 'training_history.png'))
        plt.close()
        
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
    
    def evaluate_model(self, X_test, y_test, original_prices):
        """Evaluate the model on test data and return detailed metrics"""
        if self.model is None:
            print("Model not trained or loaded")
            return None
            
        # Reshape y_test to match model output shape
        y_test_reshaped = y_test.reshape(y_test.shape[0], self.prediction_days)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Get the price index
        price_index = 0  # Assuming price is the first column in the feature set
        
        # Create arrays for inverse transformation with correct sizes
        # For predictions - one row per prediction day
        empty_array = np.zeros((len(predictions) * self.prediction_days, self.scaler.n_features_in_))
        
        # Store the predicted values for price only
        for i in range(len(predictions)):
            for j in range(self.prediction_days):
                empty_array[i*self.prediction_days + j, price_index] = predictions[i, j]
        
        # Inverse transform predictions
        predictions_original = self.scaler.inverse_transform(empty_array)[:, price_index]
        predictions_original = predictions_original.reshape(-1, self.prediction_days)
        
        # For actual values - same approach
        empty_array_actual = np.zeros((len(y_test_reshaped) * self.prediction_days, self.scaler.n_features_in_))
        
        # Store the actual values for price only
        for i in range(len(y_test_reshaped)):
            for j in range(self.prediction_days):
                empty_array_actual[i*self.prediction_days + j, price_index] = y_test_reshaped[i, j]
        
        # Inverse transform actual values
        y_test_original = self.scaler.inverse_transform(empty_array_actual)[:, price_index]
        y_test_original = y_test_original.reshape(-1, self.prediction_days)
        
        # Calculate metrics
        mse = np.mean(np.square(predictions_original - y_test_original))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_original - y_test_original))
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Plot some sample predictions
        self.plot_sample_predictions(y_test_original, predictions_original, original_prices)
        
        return predictions_original, y_test_original, rmse
    
    def plot_sample_predictions(self, actual, predicted, original_prices, num_samples=5):
        """Plot sample predictions vs actual values"""
        plt.figure(figsize=(12, 8))
        
        for i in range(min(num_samples, len(actual))):
            plt.subplot(num_samples, 1, i+1)
            plt.plot(range(self.prediction_days), actual[i], 'b-', label='Actual')
            plt.plot(range(self.prediction_days), predicted[i], 'r-', label='Predicted')
            plt.title(f'Sample {i+1}')
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'sample_predictions.png'))
        plt.close()
    
    def predict_next_days(self, recent_data):
        """Predict prices for the next 5 days using the most recent data"""
        if self.model is None:
            print("Model not trained or loaded")
            return None
        
        # Prepare recent data with all features
        if len(recent_data.shape) == 1 or recent_data.shape[1] == 1:
            # If only price data is provided, we need to generate the additional features
            price_data = recent_data.reshape(-1)
            
            # Create additional features
            price_7d_mean = np.convolve(price_data, np.ones(7)/7, mode='valid')
            price_7d_mean = np.pad(price_7d_mean, (7-1, 0), 'edge')
            
            price_30d_mean = np.convolve(price_data, np.ones(30)/30, mode='valid')
            price_30d_mean = np.pad(price_30d_mean, (30-1, 0), 'edge')
            
            # Calculate rolling standard deviation
            price_7d_std = np.array([np.std(price_data[max(0, i-6):i+1]) for i in range(len(price_data))])
            
            # Month, day_of_week, day_of_year would be added here if date information is available
            # For now, we'll just use zeros as placeholders
            additional_features = np.zeros((len(price_data), self.scaler.n_features_in_ - 4))
            
            # Combine all features
            all_features = np.column_stack([
                price_data, 
                price_7d_mean, 
                price_30d_mean, 
                price_7d_std
            ])
            
            if additional_features.shape[1] > 0:
                all_features = np.column_stack([all_features, additional_features])
        else:
            # If all features are already provided
            all_features = recent_data
            
        # Scale the features
        scaled_features = self.scaler.transform(all_features)
        
        # Create input sequence
        X_pred = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, scaled_features.shape[1])
        
        # Make prediction
        prediction = self.model.predict(X_pred)
        
        # Prepare for inverse transformation
        empty_array = np.zeros((self.prediction_days, self.scaler.n_features_in_))
        
        # Set the predicted values for price only
        price_index = 0  # Assuming price is the first column
        for i in range(self.prediction_days):
            empty_array[i, price_index] = prediction[0, i]
            
        # Inverse transform to get actual prices
        predicted_prices = self.scaler.inverse_transform(empty_array)[:, price_index]
        
        # Validate predictions
        min_price = np.min(recent_data) * 0.5
        max_price = np.max(recent_data) * 2.0
        
        # Clip predictions to reasonable range
        predicted_prices = np.clip(predicted_prices, min_price, max_price)
        
        return predicted_prices
    
    def visualize_prediction(self, actual_prices, predicted_prices, crop_name=None):
        """Visualize the actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        
        # Plot actual prices
        plt.plot(range(len(actual_prices)), actual_prices, label='Historical Prices', color='blue')
        
        # Plot predicted prices
        plt.plot(range(len(actual_prices), len(actual_prices) + len(predicted_prices)),
                 predicted_prices, label='Predicted Prices', color='red')
                 
        # Add confidence interval (simple approach)
        std_dev = np.std(actual_prices[-30:])
        upper_bound = predicted_prices + std_dev
        lower_bound = predicted_prices - std_dev
        
        plt.fill_between(
            range(len(actual_prices), len(actual_prices) + len(predicted_prices)),
            lower_bound, upper_bound, color='red', alpha=0.2, label='Confidence Interval'
        )
        
        plt.title(f'Crop Price Prediction for {crop_name if crop_name else "All Crops"}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{self.model_path}/price_prediction_{crop_name if crop_name else 'all_crops'}.png")
        plt.close()
        
        return f"{self.model_path}/price_prediction_{crop_name if crop_name else 'all_crops'}.png"
    
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
        X_train, X_test, y_train, y_test, crop_data, original_prices = self.preprocess_data(df, crop_name)
        
        # Check if model exists, if not train a new one
        if not self.load_model():
            print("Training a new model...")
            self.build_model((X_train.shape[1], X_train.shape[2]))
            self.train_model(X_train, y_train, X_test, y_test)
            self.save_model()
        
        # Evaluate the model
        predicted_prices, actual_prices, rmse = self.evaluate_model(X_test, y_test, original_prices)
        
        # Get the most recent data with all features for prediction
        if 'price_7d_mean' in crop_data.columns:
            recent_features = crop_data[['price', 'price_7d_mean', 'price_30d_mean', 'price_7d_std']].values[-self.sequence_length:]
            
            # Add additional features if available
            if 'month' in crop_data.columns:
                additional_features = crop_data[['month', 'day_of_week', 'day_of_year']].values[-self.sequence_length:]
                recent_features = np.column_stack([recent_features, additional_features])
        else:
            recent_features = crop_data['price'].values[-self.sequence_length:]
        
        # Predict next 5 days
        next_5_days_prices = self.predict_next_days(recent_features)
        
        # Get the last date in the dataset
        if 'date' in crop_data.columns:
            last_date = crop_data['date'].iloc[-1]
            future_dates = self.generate_future_dates(last_date)
            
            # Display results
            print("\nPredicted prices for the next 5 days:")
            for i, (date, price) in enumerate(zip(future_dates, next_5_days_prices)):
                print(f"Date: {date.strftime('%Y-%m-%d')}, Price: ${price:.2f}")
        else:
            # Display results without dates
            print("\nPredicted prices for the next 5 days:")
            for i, price in enumerate(next_5_days_prices):
                print(f"Day {i+1}: ${price:.2f}")
        
        # Visualize results
        plot_path = self.visualize_prediction(
            crop_data['price'].values[-60:],  # Last 60 days of actual prices
            next_5_days_prices,
            crop_name
        )
        
        print(f"\nPrediction visualization saved to: {plot_path}")
        
        return next_5_days_prices, plot_path


# Example usage
if __name__ == "__main__":
    print("Crop Price Prediction System")
    print("===========================")
    
    # Initialize the price predictor
    predictor = CropPricePredictor()
    
    # Ask for data path
    data_path = input("Enter the path to the CSV file with historical prices (default: crop_price_prediction/historical_prices.csv): ")
    if data_path.strip() == "":
        data_path = "crop_price_prediction/historical_prices.csv"
    predictor.data_path = data_path
    
    # Ask for crop name
    crop_name = input("Enter the crop name to predict (leave blank for all crops): ")
    if crop_name.strip() == "":
        crop_name = None
    
    # Make prediction
    predictor.predict_and_display(crop_name)