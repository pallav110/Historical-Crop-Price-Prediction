import os
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import traceback
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend to avoid Tkinter errors

import matplotlib.pyplot as plt


class AgriculturalAssistant:
    def __init__(self, base_path=None):
        """Initialize the integrated agricultural assistant system"""
        # Set base path with fallbacks for different environments
        if base_path is None:
            self.base_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_path = base_path
            
        # Use os.path.join for cross-platform compatibility
        self.crop_recommendation_dir = os.path.join(self.base_path, "crop_recommendation")
        self.fertilizer_recommendation_dir = os.path.join(self.base_path, "fertilizer_recommendation")
        self.crop_price_prediction_dir = os.path.join(self.base_path, "crop_price_prediction")
        
        # Create directories if they don't exist
        self._ensure_directories_exist()
        
        # Define model paths using os.path.join for cross-platform compatibility
        self.crop_model_path = os.path.join(self.crop_recommendation_dir, "best_rf_model.joblib")
        self.crop_scaler_path = os.path.join(self.crop_recommendation_dir, "scaler.joblib")
        self.crop_encoder_path = os.path.join(self.crop_recommendation_dir, "label_encoder.joblib")
        self.fertilizer_model_path = os.path.join(self.fertilizer_recommendation_dir, "fertilizer_model.pkl")
        self.fertilizer_encoder_path = os.path.join(self.fertilizer_recommendation_dir, "label_encoders.pkl")
        self.price_model_path = os.path.join(self.crop_price_prediction_dir, "lstm_model.h5")
        self.price_scaler_path = os.path.join(self.crop_price_prediction_dir, "price_scaler.pkl")
        
        # Define output directories
        self.output_dir = os.path.join(self.base_path, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models and related components
        self.crop_model = None
        self.crop_scaler = None
        self.crop_encoder = None
        self.fertilizer_model = None
        self.fertilizer_encoders = None
        self.price_model = None
        self.price_scaler = None
        
        # Load models
        self.load_models()
        
        # Fertilizer recommendation dictionary - more balanced thresholds
        self.fertilizer_dic = {
            'NHigh': """The N value of soil is high and might give rise to weeds.
                 Suggestions:
                1. Add manure
                2. Use coffee grounds
                3. Plant nitrogen-fixing plants
                4. Plant green manure crops like cabbage
                5. Use mulch""",
            
            'Nlow': """The N value of your soil is low.
                 Suggestions:
                1. Add sawdust or woodchips
                2. Plant heavy nitrogen-feeding plants
                3. Add composted manure
                4. Use NPK fertilizers with high N value""",
            
            'PHigh': """The P value of your soil is high.
                 Suggestions:
                1. Avoid adding manure
                2. Use phosphorus-free fertilizers
                3. Plant nitrogen-fixing vegetables""",
            
            'Plow': """The P value of your soil is low.
                 Suggestions:
                1. Add bone meal or rock phosphate
                2. Use fertilizers with high P values
                3. Add organic compost""",
            
            'KHigh': """The K value of your soil is high.
                 Suggestions:
                1. Loosen soil and water thoroughly
                2. Remove potassium-rich rocks
                3. Use fertilizers with low K values""",
            
            'Klow': """The K value of your soil is low.
                 Suggestions:
                1. Add muricate or sulphate of potash
                2. Use potash fertilizers
                3. Bury banana peels below soil surface"""
        }
        
    def _ensure_directories_exist(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.crop_recommendation_dir, 
                          self.fertilizer_recommendation_dir, 
                          self.crop_price_prediction_dir]:
            os.makedirs(directory, exist_ok=True)
        
    def load_models(self):
        """Load all models if they exist with better error handling"""
        # Load crop recommendation model
        if all(os.path.exists(path) for path in [self.crop_model_path, self.crop_scaler_path, self.crop_encoder_path]):
            try:
                print("Loading crop recommendation model...")
                self.crop_model = joblib.load(self.crop_model_path)
                self.crop_scaler = joblib.load(self.crop_scaler_path)
                self.crop_encoder = joblib.load(self.crop_encoder_path)
                print("Crop recommendation model loaded successfully.")
            except Exception as e:
                print(f"Error loading crop recommendation model: {e}")
        else:
            print("Crop recommendation model not found. Some features will be disabled.")
            
        # Load fertilizer recommendation model
        if all(os.path.exists(path) for path in [self.fertilizer_model_path, self.fertilizer_encoder_path]):
            try:
                print("Loading fertilizer recommendation model...")
                self.fertilizer_model = joblib.load(self.fertilizer_model_path)
                with open(self.fertilizer_encoder_path, "rb") as le_file:
                    self.fertilizer_encoders = pickle.load(le_file)
                print("Fertilizer recommendation model loaded successfully.")
            except Exception as e:
                print(f"Error loading fertilizer recommendation model: {e}")
        else:
            print("Fertilizer recommendation model not found. Some features will be disabled.")
            
        # Load price prediction model
        if all(os.path.exists(path) for path in [self.price_model_path, self.price_scaler_path]):
            try:
                print("Loading price prediction model...")
                self.price_model = tf.keras.models.load_model(self.price_model_path)
                self.price_scaler = joblib.load(self.price_scaler_path)
                print("Price prediction model loaded successfully.")
            except Exception as e:
                print(f"Error loading price prediction model: {e}")
        else:
            print("Price prediction model not found. Some features will be disabled.")
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """Recommend a crop based on soil parameters and environmental conditions"""
        if self.crop_model is None or self.crop_scaler is None or self.crop_encoder is None:
            return "Crop recommendation model not loaded."
            
        try:
            # Validate input ranges 
            if not (0 <= N <= 200 and 0 <= P <= 200 and 0 <= K <= 200 and
                   -20 <= temperature <= 60 and 0 <= humidity <= 100 and
                   0 <= ph <= 14 and 0 <= rainfall <= 5000):
                return "Input values are outside of expected ranges. Please check your values."
                
            # Prepare input data
            user_input = [[N, P, K, temperature, humidity, ph, rainfall]]
            user_input_scaled = self.crop_scaler.transform(user_input)
            
            # Predict the crop
            prediction = self.crop_model.predict(user_input_scaled)
            predicted_crop = self.crop_encoder.inverse_transform(prediction)[0]
            
            return predicted_crop
        except Exception as e:
            return f"Error in crop recommendation: {e}"
    
    def recommend_fertilizer(self, temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
        """Recommend fertilizer based on soil parameters and crop type"""
        if self.fertilizer_model is None or self.fertilizer_encoders is None:
            return "Fertilizer recommendation model not loaded."
            
        try:
            # Validate input ranges
            if not (0 <= temperature <= 50 and 0 <= humidity <= 100 and 
                    0 <= moisture <= 100 and 0 <= nitrogen <= 200 and 
                    0 <= potassium <= 200 and 0 <= phosphorous <= 200):
                return "Input values are outside of expected ranges. Please check your values."
                
            # Check if the soil type and crop type are in the trained vocabulary
            if 'Soil Type' not in self.fertilizer_encoders or 'Crop Type' not in self.fertilizer_encoders:
                return "Error: Required label encoders not found in the model."
                
            soil_types = self.fertilizer_encoders['Soil Type'].classes_
            crop_types = self.fertilizer_encoders['Crop Type'].classes_
            
            # Case-insensitive matching for user convenience
            soil_type_lower = soil_type.lower()
            crop_type_lower = crop_type.lower()
            
            soil_match = next((s for s in soil_types if s.lower() == soil_type_lower), None)
            crop_match = next((c for c in crop_types if c.lower() == crop_type_lower), None)
            
            if not soil_match:
                valid_soils = ", ".join(soil_types)
                return f"Invalid soil type. Please use one of: {valid_soils}"
                
            if not crop_match:
                valid_crops = ", ".join(crop_types)
                return f"Invalid crop type. Please use one of: {valid_crops}"
            
            # Encode categorical inputs using the matched values
            soil_type_encoded = self.fertilizer_encoders['Soil Type'].transform([soil_match])[0]
            crop_type_encoded = self.fertilizer_encoders['Crop Type'].transform([crop_match])[0]
            
            # Prepare input data
            input_data = [[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, 
                          nitrogen, potassium, phosphorous]]
            
            # Predict fertilizer
            fertilizer_code = self.fertilizer_model.predict(input_data)[0]
            fertilizer_name = self.fertilizer_encoders['Fertilizer Name'].inverse_transform([fertilizer_code])[0]
            
            # Get nutrient suggestions based on crop-specific thresholds
            suggestions = self.get_nutrient_suggestions(nitrogen, potassium, phosphorous, crop_match)
            
            return {
                "fertilizer": fertilizer_name,
                "suggestions": suggestions
            }
        except Exception as e:
            traceback.print_exc()  # Print full traceback for debugging
            return f"Error in fertilizer recommendation: {e}"
    
    def get_nutrient_suggestions(self, nitrogen, potassium, phosphorous, crop_type="general"):
        """Get suggestions based on nutrient levels with crop-specific thresholds"""
        # Define crop-specific thresholds (simplified example)
        crop_thresholds = {
            "rice": {"n_high": 120, "n_low": 60, "p_high": 80, "p_low": 30, "k_high": 80, "k_low": 30},
            "wheat": {"n_high": 100, "n_low": 40, "p_high": 70, "p_low": 20, "k_high": 70, "k_low": 25},
            "maize": {"n_high": 150, "n_low": 80, "p_high": 100, "p_low": 40, "k_high": 100, "k_low": 40},
            # Default thresholds for general case or unknown crops
            "general": {"n_high": 80, "n_low": 40, "p_high": 60, "p_low": 20, "k_high": 60, "k_low": 20}
        }
        
        # Use general thresholds if crop not found
        thresholds = crop_thresholds.get(crop_type.lower() if isinstance(crop_type, str) else "general", 
                                        crop_thresholds["general"])
        
        suggestions = []
        if nitrogen > thresholds["n_high"]:
            suggestions.append(self.fertilizer_dic['NHigh'])
        elif nitrogen < thresholds["n_low"]:
            suggestions.append(self.fertilizer_dic['Nlow'])
        
        if phosphorous > thresholds["p_high"]:
            suggestions.append(self.fertilizer_dic['PHigh'])
        elif phosphorous < thresholds["p_low"]:
            suggestions.append(self.fertilizer_dic['Plow'])
        
        if potassium > thresholds["k_high"]:
            suggestions.append(self.fertilizer_dic['KHigh'])
        elif potassium < thresholds["k_low"]:
            suggestions.append(self.fertilizer_dic['Klow'])
        
        # Return "balanced" if no specific suggestions
        if not suggestions:
            suggestions.append("Your soil nutrient levels appear balanced for this crop type.")
            
        return suggestions
    
    def predict_crop_prices(self, crop_name, historical_data_path):
        """Predict crop prices for the next 5 days with improved reliability and error handling"""
        if self.price_model is None or self.price_scaler is None:
            return "Price prediction model not loaded."
            
        try:
            # Check if file exists with absolute/relative path handling
            if not os.path.isabs(historical_data_path):
                # Try relative to current directory first
                if os.path.exists(historical_data_path):
                    pass  # File exists as provided
                elif os.path.exists(os.path.join(self.base_path, historical_data_path)):
                    historical_data_path = os.path.join(self.base_path, historical_data_path)
                else:
                    return f"Error: Historical data file not found at {historical_data_path}"
            
            # Load historical price data
            try:
                df = pd.read_csv(historical_data_path)
            except Exception as e:
                return f"Error loading data file: {e}"
                
            # Basic data validation
            if 'price' not in df.columns:
                return "Error: Price column missing in historical data"
            
            # Filter by crop name if provided
            if crop_name and 'crop_name' in df.columns:
                crop_data = df[df['crop_name'] == crop_name].copy()  # Use copy to avoid SettingWithCopyWarning
                if crop_data.empty:
                    # Try case-insensitive matching
                    crop_match = df['crop_name'].str.lower() == crop_name.lower()
                    if crop_match.any():
                        crop_data = df[crop_match].copy()
                    else:
                        available_crops = ", ".join(df['crop_name'].unique())
                        return f"No data found for crop: {crop_name}. Available crops: {available_crops}"
            else:
                crop_data = df.copy()
            
            # Ensure date column exists and is properly formatted
            if 'date' in crop_data.columns:
                try:
                    crop_data['date'] = pd.to_datetime(crop_data['date'])
                    crop_data = crop_data.sort_values('date')
                except Exception as e:
                    return f"Error converting date column: {e}"
            
            # Check for and handle missing values
            if crop_data['price'].isnull().any():
                print("Warning: Prices contain missing values. Filling with forward and backward fill.")
                crop_data['price'] = crop_data['price'].ffill().bfill()
            
            # Check if we have sufficient data
            sequence_length = 30  # Must match the model's expected sequence length
            if len(crop_data) < sequence_length:
                return f"Not enough historical data. Need at least {sequence_length} days, but found {len(crop_data)}."
            
            # Extract price data
            prices = crop_data['price'].values
            
            # Verify price data has valid values
            if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
                return "Error: Price data contains NaN or infinite values after preprocessing"
                
            # Handle negative or zero prices
            if np.any(prices <= 0):
                min_price = np.min(prices)
                if min_price <= 0:
                    print(f"Warning: Found non-positive prices (minimum: {min_price}). Adding offset.")
                    offset = abs(min_price) + 0.01
                    prices = prices + offset
            
            try:
                # Create additional features similar to what was used during training
                # Calculate rolling statistics for price
                price_7d_mean = np.convolve(prices, np.ones(7)/7, mode='same')
                price_30d_mean = np.convolve(prices, np.ones(30)/30, mode='same')
                
                # Calculate rolling standard deviation (simple approach)
                price_7d_std = []
                for i in range(len(prices)):
                    start_idx = max(0, i-6)
                    price_7d_std.append(np.std(prices[start_idx:i+1]))
                price_7d_std = np.array(price_7d_std)
                
                # Get date features if available
                if 'date' in crop_data.columns:
                    month = crop_data['date'].dt.month.values
                    day_of_week = crop_data['date'].dt.dayofweek.values
                    day_of_year = crop_data['date'].dt.dayofyear.values
                else:
                    # Use dummy values if date not available
                    month = np.ones_like(prices)
                    day_of_week = np.ones_like(prices)
                    day_of_year = np.ones_like(prices)
                
                # Combine all features - must match exactly what the model was trained with
                features = np.column_stack([
                    prices,
                    price_7d_mean,
                    price_30d_mean,
                    price_7d_std,
                    month,
                    day_of_week,
                    day_of_year
                ])
                
                # Create input sequence for prediction (use last sequence_length days)
                X_pred = features[-sequence_length:].reshape(1, sequence_length, 7)
                
                # Scale the features
                X_pred_scaled = np.zeros_like(X_pred)
                for i in range(sequence_length):
                    X_pred_scaled[0, i] = self.price_scaler.transform(X_pred[0, i].reshape(1, -1))
                
                # Make prediction
                prediction_scaled = self.price_model.predict(X_pred_scaled)
                
                # Reshape prediction for inverse transform
                prediction_scaled_reshaped = prediction_scaled.reshape(-1, 1)
                
                # Inverse transform to get actual prices
                # Create dummy array with same number of features
                dummy_array = np.zeros((len(prediction_scaled_reshaped), 7))
                dummy_array[:, 0] = prediction_scaled_reshaped.flatten()  # Put scaled predictions in price column
                
                # Inverse transform and extract only the price column
                predicted_full = self.price_scaler.inverse_transform(dummy_array)
                predicted_prices = predicted_full[:, 0]  # Extract price column
                
                # Post-process predictions to ensure they're reasonable
                # Clip to reasonable range based on historical data
                min_allowed = max(0.1, np.min(prices) * 0.5)  # Min price at least 0.1 but could be lower if data shows it
                max_allowed = np.max(prices) * 2.0  # Max at double the highest historical price
                predicted_prices = np.clip(predicted_prices, min_allowed, max_allowed)
                
                # Generate future dates
                prediction_days = len(predicted_prices)
                if 'date' in crop_data.columns:
                    last_date = crop_data['date'].iloc[-1]
                    future_dates = self.generate_future_dates(last_date, prediction_days)
                    
                    # Create result dictionary
                    result = {
                        "prediction": [
                            {"date": date.strftime('%Y-%m-%d'), "price": float(price)}
                            for date, price in zip(future_dates, predicted_prices)
                        ]
                    }
                else:
                    result = {
                        "prediction": [
                            {"day": i+1, "price": float(price)}
                            for i, price in enumerate(predicted_prices)
                        ]
                    }
                
                # Visualize the prediction
                plot_file = self.visualize_price_prediction(
                    crop_data['price'].values[-30:],
                    predicted_prices,
                    crop_name
                )
                
                result["plot_file"] = plot_file
                return result
                
            except Exception as e:
                traceback.print_exc()  # Print full stack trace for debugging
                return f"Error during prediction: {e}"
                
        except Exception as e:
            traceback.print_exc()  # Print full stack trace for debugging
            return f"Error in price prediction: {str(e)}"
    
    def generate_future_dates(self, last_date, days=5):
        """Generate dates for the next given number of days starting from last_date"""
        future_dates = []
        current_date = last_date
        
        for _ in range(days):
            current_date += timedelta(days=1)
            future_dates.append(current_date)
            
        return future_dates
    
    def visualize_price_prediction(self, actual_prices, predicted_prices, crop_name=None):
        """Visualize the actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        
        # Plot actual prices
        plt.plot(range(len(actual_prices)), actual_prices, label='Historical Prices', color='blue')
        
        # Plot predicted prices
        plt.plot(range(len(actual_prices), len(actual_prices) + len(predicted_prices)),
                 predicted_prices, label='Predicted Prices', color='red')
        
        # Add confidence interval
        std_dev = np.std(actual_prices)
        plt.fill_between(
            range(len(actual_prices), len(actual_prices) + len(predicted_prices)),
            predicted_prices - std_dev, predicted_prices + std_dev,
            color='red', alpha=0.2, label='Confidence Interval (±1σ)'
        )
                 
        plt.title(f'Crop Price Prediction for {crop_name if crop_name else "All Crops"}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save the plot to output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"price_prediction_{crop_name if crop_name else 'all_crops'}_{timestamp}.png"
        output_file = os.path.join(self.output_dir, filename)
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def run_interactive(self):
        """Run the agricultural assistant in interactive mode with improved UI"""
        while True:
            print("\n===== Agricultural Assistant =====")
            print("1. Crop Recommendation")
            print("2. Fertilizer Recommendation")
            print("3. Crop Price Prediction")
            print("4. Exit")
            
            try:
                choice = input("\nEnter your choice (1-4): ")
                
                if choice == '1':
                    self.run_crop_recommendation()
                elif choice == '2':
                    self.run_fertilizer_recommendation()
                elif choice == '3':
                    self.run_price_prediction()
                elif choice == '4':
                    print("Thank you for using the Agricultural Assistant!")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except KeyboardInterrupt:
                print("\nProgram interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                print("Please try again.")
    
    def run_crop_recommendation(self):
        """Interactive crop recommendation with improved input validation"""
        print("\n----- Crop Recommendation -----")
        
        if self.crop_model is None:
            print("Crop recommendation model is not loaded. This feature is unavailable.")
            return
            
        try:
            # Input with validation
            try:
                N = self._get_float_input("Nitrogen content (N) [0-200]: ", 0, 200)
                P = self._get_float_input("Phosphorus content (P) [0-200]: ", 0, 200)
                K = self._get_float_input("Potassium content (K) [0-200]: ", 0, 200)
                temperature = self._get_float_input("Temperature (°C) [-20-60]: ", -20, 60)
                humidity = self._get_float_input("Humidity (%) [0-100]: ", 0, 100)
                ph = self._get_float_input("pH level [0-14]: ", 0, 14)
                rainfall = self._get_float_input("Rainfall (mm) [0-5000]: ", 0, 5000)
            except KeyboardInterrupt:
                print("\nInput cancelled.")
                return
                
            crop = self.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
            
            if isinstance(crop, str) and "Error" in crop:
                print(f"\n{crop}")
            else:
                print(f"\nRecommended Crop: {crop}")
                
                # Offer information about the recommended crop
                print("\nWould you like to learn more about growing this crop?")
                more_info = input("Enter 'y' for Yes or any other key to return to main menu: ")
                if more_info.lower() == 'y':
                    print(f"\nBasic information about growing {crop}:")
                    print("This would display growing tips and requirements for the specific crop.")
                    print("Feature to be implemented in a future update.")
                    
        except ValueError as e:
            print(f"Error: {e}. Please enter valid numeric values.")
    
    def run_fertilizer_recommendation(self):
        """Interactive fertilizer recommendation with improved input validation"""
        print("\n----- Fertilizer Recommendation -----")
        
        if self.fertilizer_model is None:
            print("Fertilizer recommendation model is not loaded. This feature is unavailable.")
            return
            
        try:
            # Input with validation
            try:
                temperature = self._get_float_input("Temperature (°C) [0-50]: ", 0, 50)
                humidity = self._get_float_input("Humidity (%) [0-100]: ", 0, 100)
                moisture = self._get_float_input("Moisture (%) [0-100]: ", 0, 100)
                
                # Show available options
                if self.fertilizer_encoders and 'Soil Type' in self.fertilizer_encoders:
                    soil_types = ", ".join(self.fertilizer_encoders['Soil Type'].classes_)
                    print(f"Available soil types: {soil_types}")
                soil_type = input("Soil Type: ")
                
                if self.fertilizer_encoders and 'Crop Type' in self.fertilizer_encoders:
                    crop_types = ", ".join(self.fertilizer_encoders['Crop Type'].classes_)
                    print(f"Available crop types: {crop_types}")
                crop_type = input("Crop Type: ")
                
                nitrogen = self._get_int_input("Nitrogen (N) value [0-200]: ", 0, 200)
                potassium = self._get_int_input("Potassium (K) value [0-200]: ", 0, 200)
                phosphorous = self._get_int_input("Phosphorous (P) value [0-200]: ", 0, 200)
            except KeyboardInterrupt:
                print("\nInput cancelled.")
                return
                
            result = self.recommend_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous)
            
            if isinstance(result, dict):
                print(f"\nRecommended Fertilizer: {result['fertilizer']}")
                print("\nNutrient Suggestions:")
                for suggestion in result['suggestions']:
                    print(suggestion)
            else:
                print(result)  # Error message
        except ValueError as e:
            print(f"Error: {e}. Please enter valid values.")
    
    def run_price_prediction(self):
        """Interactive crop price prediction with improved file handling"""
        print("\n----- Crop Price Prediction -----")
        
        if self.price_model is None:
            print("Price prediction model is not loaded. This feature is unavailable.")
            return
            
        try:
            # Let user specify the data file with default option
            default_path = os.path.join(self.crop_price_prediction_dir, "historical_prices.csv")
            prompt = f"Enter path to historical prices CSV file\n(default: {default_path}): "
            
            historical_data_path = input(prompt).strip()
            if not historical_data_path:
                historical_data_path = default_path
                
            # Check if file exists
            if not os.path.exists(historical_data_path):
                # Try relative to base path
                alt_path = os.path.join(self.base_path, historical_data_path)
                if os.path.exists(alt_path):
                    historical_data_path = alt_path
                else:
                    print(f"Error: File not found at {historical_data_path}")
                    return
                    
            # Preview available crops
            try:
                df = pd.read_csv(historical_data_path)
                if 'crop_name' in df.columns:
                    available_crops = df['crop_name'].unique()
                    print(f"Available crops in dataset: {', '.join(available_crops)}")
            except Exception as e:
                print(f"Warning: Could not preview crops in file: {e}")
                
            crop_name = input("Enter crop name (leave blank for all crops): ")
            if crop_name.strip() == "":
                crop_name = None
                
            print("\nPredicting prices. This may take a moment...")
            result = self.predict_crop_prices(crop_name, historical_data_path)
            
            if isinstance(result, dict) and "prediction" in result:
                print("\nPredicted prices for the next 5 days:")
                for day in result['prediction']:
                    if 'date' in day:
                        print(f"Date: {day['date']}, Price: ${day['price']:.2f}")
                    else:
                        print(f"Day {day['day']}: ${day['price']:.2f}")
                        
                if "plot_file" in result:
                    print(f"\nA visualization has been saved to: {result['plot_file']}")
                    
                    # If on a system that supports it, offer to open the plot
                    if sys.platform.startswith('darwin'):  # macOS
                        open_plot = input("\nWould you like to open the plot? (y/n): ")
                        if open_plot.lower() == 'y':
                            os.system(f"open {result['plot_file']}")
                    elif sys.platform.startswith('win'):  # Windows
                        open_plot = input("\nWould you like to open the plot? (y/n): ")
                        if open_plot.lower() == 'y':
                            os.system(f"start {result['plot_file']}")
                    elif sys.platform.startswith('linux'):  # Linux
                        open_plot = input("\nWould you like to open the plot? (y/n): ")
                        if open_plot.lower() == 'y':
                            os.system(f"xdg-open {result['plot_file']}")
            else:
                print(f"\n{result}")  # Error message
        except ValueError as e:
            print(f"Error: {e}. Please enter valid values.")
    
    def _get_float_input(self, prompt, min_val=None, max_val=None):
        """Get float input with validation"""
        while True:
            try:
                value = float(input(prompt))
                if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                    if min_val is not None and max_val is not None:
                        print(f"Value must be between {min_val} and {max_val}. Please try again.")
                    elif min_val is not None:
                        print(f"Value must be at least {min_val}. Please try again.")
                    elif max_val is not None:
                        print(f"Value must be at most {max_val}. Please try again.")
                else:
                    return value
            except ValueError:
                print("Please enter a valid number.")
    
    def _get_int_input(self, prompt, min_val=None, max_val=None):
        """Get integer input with validation"""
        while True:
            try:
                value = int(input(prompt))
                if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                    if min_val is not None and max_val is not None:
                        print(f"Value must be between {min_val} and {max_val}. Please try again.")
                    elif min_val is not None:
                        print(f"Value must be at least {min_val}. Please try again.")
                    elif max_val is not None:
                        print(f"Value must be at most {max_val}. Please try again.")
                else:
                    return value
            except ValueError:
                print("Please enter a valid integer.")
    
    def generate_report(self, crop_name, soil_data, weather_data=None, price_data_path=None):
        """Generate a comprehensive report for a specific crop"""
        report = {
            "crop_name": crop_name,
            "soil_analysis": {},
            "recommendations": {},
            "price_forecast": {}
        }
        
        try:
            # Parse soil data
            N = soil_data.get("nitrogen", 0)
            P = soil_data.get("phosphorous", 0)
            K = soil_data.get("potassium", 0)
            ph = soil_data.get("ph", 7.0)
            
            # Add soil analysis to report
            report["soil_analysis"] = {
                "nitrogen": N,
                "phosphorous": P,
                "potassium": K,
                "ph": ph
            }
            
            # Get weather data if available
            temperature = weather_data.get("temperature", 25) if weather_data else 25
            humidity = weather_data.get("humidity", 60) if weather_data else 60
            rainfall = weather_data.get("rainfall", 200) if weather_data else 200
            moisture = weather_data.get("moisture", 50) if weather_data else 50
            
            # Check if this crop is suitable
            if self.crop_model is not None:
                recommended_crop = self.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
                suitability = "Highly Suitable" if recommended_crop == crop_name else "May Not Be Optimal"
                report["recommendations"]["crop_suitability"] = suitability
                
                if recommended_crop != crop_name:
                    report["recommendations"]["alternative_crop"] = recommended_crop
            
            # Get fertilizer recommendation
            if self.fertilizer_model is not None:
                soil_type = soil_data.get("soil_type", "Loamy")  # Default to Loamy if not specified
                fertilizer_result = self.recommend_fertilizer(
                    temperature, humidity, moisture, soil_type, crop_name, N, P, K
                )
                
                if isinstance(fertilizer_result, dict):
                    report["recommendations"]["fertilizer"] = fertilizer_result["fertilizer"]
                    report["recommendations"]["nutrient_suggestions"] = fertilizer_result["suggestions"]
            
            # Get price forecast if price data is available
            if self.price_model is not None and price_data_path:
                price_result = self.predict_crop_prices(crop_name, price_data_path)
                
                if isinstance(price_result, dict) and "prediction" in price_result:
                    report["price_forecast"]["predictions"] = price_result["prediction"]
                    report["price_forecast"]["plot_file"] = price_result.get("plot_file", "")
            
            return report
            
        except Exception as e:
            return f"Error generating report: {e}"

    def export_report(self, report, output_format="text"):
        """Export the report in the specified format (text, json, csv)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agricultural_report_{timestamp}"
        
        try:
            if output_format == "json":
                import json
                output_file = os.path.join(self.output_dir, f"{filename}.json")
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=4)
                    
            elif output_format == "csv":
                output_file = os.path.join(self.output_dir, f"{filename}.csv")
                
                # Flatten the report structure for CSV
                flat_data = []
                crop_name = report["crop_name"]
                
                # Add soil analysis
                for key, value in report["soil_analysis"].items():
                    flat_data.append({"crop_name": crop_name, "category": "soil_analysis", 
                                      "attribute": key, "value": value})
                
                # Add recommendations
                for key, value in report["recommendations"].items():
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            flat_data.append({"crop_name": crop_name, "category": "recommendations", 
                                              "attribute": f"{key}_{i+1}", "value": item})
                    else:
                        flat_data.append({"crop_name": crop_name, "category": "recommendations", 
                                          "attribute": key, "value": value})
                
                # Add price forecast if available
                if "predictions" in report["price_forecast"]:
                    for i, pred in enumerate(report["price_forecast"]["predictions"]):
                        flat_data.append({"crop_name": crop_name, "category": "price_forecast", 
                                          "attribute": f"day_{i+1}", "value": pred["price"]})
                
                # Write to CSV
                pd.DataFrame(flat_data).to_csv(output_file, index=False)
                
            else:  # Default to text format
                output_file = os.path.join(self.output_dir, f"{filename}.txt")
                with open(output_file, "w") as f:
                    f.write(f"AGRICULTURAL ANALYSIS REPORT\n")
                    f.write(f"=========================\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Crop: {report['crop_name']}\n\n")
                    
                    # Soil Analysis
                    f.write("SOIL ANALYSIS\n")
                    f.write("------------\n")
                    for key, value in report["soil_analysis"].items():
                        f.write(f"{key.capitalize()}: {value}\n")
                    f.write("\n")
                    
                    # Recommendations
                    f.write("RECOMMENDATIONS\n")
                    f.write("--------------\n")
                    for key, value in report["recommendations"].items():
                        if isinstance(value, list):
                            f.write(f"{key.replace('_', ' ').capitalize()}:\n")
                            for item in value:
                                f.write(f"- {item}\n")
                        else:
                            f.write(f"{key.replace('_', ' ').capitalize()}: {value}\n")
                    f.write("\n")
                    
                    # Price Forecast
                    if "predictions" in report["price_forecast"]:
                        f.write("PRICE FORECAST\n")
                        f.write("-------------\n")
                        for pred in report["price_forecast"]["predictions"]:
                            if "date" in pred:
                                f.write(f"Date: {pred['date']}, Price: Rupees{pred['price']:.2f}\n")
                            else:
                                f.write(f"Day {pred['day']}: Rupees{pred['price']:.2f}\n")
                        
                        if "plot_file" in report["price_forecast"]:
                            f.write(f"\nPrice forecast plot saved to: {report['price_forecast']['plot_file']}\n")
            
            return f"Report successfully exported to {output_file}"
            
        except Exception as e:
            return f"Error exporting report: {e}"


# Example usage
if __name__ == "__main__":
    assistant = AgriculturalAssistant()
    assistant.run_interactive()