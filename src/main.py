import os
import pandas as pd
import joblib
import pickle
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

class AgriculturalAssistant:
    def __init__(self):
        """Initialize the integrated agricultural assistant system"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.crop_model_path = os.path.join(self.base_path, "best_rf_model.joblib")
        self.crop_scaler_path = os.path.join(self.base_path, "scaler.joblib")
        self.crop_encoder_path = os.path.join(self.base_path, "label_encoder.joblib")
        self.fertilizer_model_path = os.path.join(self.base_path, "fertilizer_model.pkl")
        self.fertilizer_encoder_path = os.path.join(self.base_path, "label_encoders.pkl")
        self.price_model_path = os.path.join(self.base_path, "crop_price_model/lstm_model.h5")
        self.price_scaler_path = os.path.join(self.base_path, "crop_price_model/price_scaler.pkl")
        
        # Load models if they exist
        self.crop_model = None
        self.crop_scaler = None
        self.crop_encoder = None
        self.fertilizer_model = None
        self.fertilizer_encoders = None
        self.price_model = None
        self.price_scaler = None
        
        self.load_models()
        
        # Fertilizer recommendation dictionary
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
        
    def load_models(self):
        """Load all models if they exist"""
        # Load crop recommendation model
        if os.path.exists(self.crop_model_path) and os.path.exists(self.crop_scaler_path) and os.path.exists(self.crop_encoder_path):
            print("Loading crop recommendation model...")
            self.crop_model = joblib.load(self.crop_model_path)
            self.crop_scaler = joblib.load(self.crop_scaler_path)
            self.crop_encoder = joblib.load(self.crop_encoder_path)
        else:
            print("Crop recommendation model not found.")
            
        # Load fertilizer recommendation model
        if os.path.exists(self.fertilizer_model_path) and os.path.exists(self.fertilizer_encoder_path):
            print("Loading fertilizer recommendation model...")
            self.fertilizer_model = joblib.load(self.fertilizer_model_path)
            with open(self.fertilizer_encoder_path, "rb") as le_file:
                self.fertilizer_encoders = pickle.load(le_file)
        else:
            print("Fertilizer recommendation model not found.")
            
        # Load price prediction model
        if os.path.exists(self.price_model_path) and os.path.exists(self.price_scaler_path):
            print("Loading price prediction model...")
            self.price_model = tf.keras.models.load_model(self.price_model_path)
            self.price_scaler = joblib.load(self.price_scaler_path)
        else:
            print("Price prediction model not found.")
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """Recommend a crop based on soil parameters and environmental conditions"""
        if self.crop_model is None or self.crop_scaler is None or self.crop_encoder is None:
            return "Crop recommendation model not loaded."
            
        try:
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
            # Encode categorical inputs
            try:
                soil_type_encoded = self.fertilizer_encoders['Soil Type'].transform([soil_type])[0]
                crop_type_encoded = self.fertilizer_encoders['Crop Type'].transform([crop_type])[0]
            except ValueError:
                return "Invalid soil type or crop type. Please check your input."
            
            # Prepare input data
            input_data = [[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]]
            
            # Predict fertilizer
            fertilizer_code = self.fertilizer_model.predict(input_data)[0]
            fertilizer_name = self.fertilizer_encoders['Fertilizer Name'].inverse_transform([fertilizer_code])[0]
            
            # Get nutrient suggestions
            suggestions = self.get_nutrient_suggestions(nitrogen, potassium, phosphorous)
            
            return {
                "fertilizer": fertilizer_name,
                "suggestions": suggestions
            }
        except Exception as e:
            return f"Error in fertilizer recommendation: {e}"
    
    def get_nutrient_suggestions(self, nitrogen, potassium, phosphorous):
        """Get suggestions based on nutrient levels"""
        suggestions = []
        if nitrogen > 80:  # Example threshold
            suggestions.append(self.fertilizer_dic['NHigh'])
        elif nitrogen < 20:  # Example threshold
            suggestions.append(self.fertilizer_dic['Nlow'])
        
        if phosphorous > 60:  # Example threshold
            suggestions.append(self.fertilizer_dic['PHigh'])
        elif phosphorous < 20:  # Example threshold
            suggestions.append(self.fertilizer_dic['Plow'])
        
        if potassium > 60:  # Example threshold
            suggestions.append(self.fertilizer_dic['KHigh'])
        elif potassium < 20:  # Example threshold
            suggestions.append(self.fertilizer_dic['Klow'])
        
        return suggestions
    
    def predict_crop_prices(self, crop_name, historical_data_path):
        """Predict crop prices for the next 5 days"""
        if self.price_model is None or self.price_scaler is None:
            return "Price prediction model not loaded."
            
        try:
            # Load historical price data
            df = pd.read_csv(historical_data_path)
            
            # Filter by crop name if provided
            if crop_name and 'crop_name' in df.columns:
                crop_data = df[df['crop_name'] == crop_name]
                if crop_data.empty:
                    return f"No data found for crop: {crop_name}"
            else:
                crop_data = df
            
            # Ensure date column exists and is properly formatted
            if 'date' in crop_data.columns:
                crop_data['date'] = pd.to_datetime(crop_data['date'])
                crop_data = crop_data.sort_values('date')
            
            # Extract and scale price data
            prices = crop_data['price'].values.reshape(-1, 1)
            scaled_prices = self.price_scaler.transform(prices)
            
            # Prepare input for prediction (use last 30 days)
            sequence_length = 30
            prediction_days = 5
            
            if len(scaled_prices) < sequence_length:
                return f"Not enough historical data. Need at least {sequence_length} days."
            
            X_pred = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
            
            # Make prediction
            prediction = self.price_model.predict(X_pred)
            
            # Inverse transform to get actual prices
            predicted_prices = self.price_scaler.inverse_transform(prediction)[0]
            
            # Generate future dates
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
            self.visualize_price_prediction(
                crop_data['price'].values[-30:],
                predicted_prices,
                crop_name
            )
            
            return result
            
        except Exception as e:
            return f"Error in price prediction: {e}"
    
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
                 
        plt.title(f'Crop Price Prediction for {crop_name if crop_name else "All Crops"}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_file = f"price_prediction_{crop_name if crop_name else 'all_crops'}.png"
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def run_interactive(self):
        """Run the agricultural assistant in interactive mode"""
        print("\n===== Agricultural Assistant =====")
        print("1. Crop Recommendation")
        print("2. Fertilizer Recommendation")
        print("3. Crop Price Prediction")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            self.run_crop_recommendation()
        elif choice == '2':
            self.run_fertilizer_recommendation()
        elif choice == '3':
            self.run_price_prediction()
        elif choice == '4':
            print("Thank you for using the Agricultural Assistant!")
            return
        else:
            print("Invalid choice. Please try again.")
        
        # Return to main menu
        self.run_interactive()
    
    def run_crop_recommendation(self):
        """Interactive crop recommendation"""
        print("\n----- Crop Recommendation -----")
        try:
            N = float(input("Nitrogen content (N): "))
            P = float(input("Phosphorus content (P): "))
            K = float(input("Potassium content (K): "))
            temperature = float(input("Temperature (°C): "))
            humidity = float(input("Humidity (%): "))
            ph = float(input("pH level: "))
            rainfall = float(input("Rainfall (mm): "))
            
            crop = self.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
            print(f"\nRecommended Crop: {crop}")
        except ValueError as e:
            print(f"Error: {e}. Please enter valid numeric values.")
    
    def run_fertilizer_recommendation(self):
            """Interactive fertilizer recommendation"""
            print("\n----- Fertilizer Recommendation -----")
            try:
                temperature = float(input("Temperature (°C): "))
                humidity = float(input("Humidity (%): "))
                moisture = float(input("Moisture (%): "))
                soil_type = input("Soil Type (e.g., Loamy, Sandy, Clayey): ")
                crop_type = input("Crop Type (e.g., Maize, Rice, Wheat): ")
                nitrogen = int(input("Nitrogen (N) value: "))
                potassium = int(input("Potassium (K) value: "))
                phosphorous = int(input("Phosphorous (P) value: "))
                
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
        """Interactive crop price prediction"""
        print("\n----- Crop Price Prediction -----")
        try:
            historical_data_path = input("Enter path to historical price data CSV: ")
            if not os.path.exists(historical_data_path):
                print(f"Error: File not found at {historical_data_path}")
                return
                
            crop_name = input("Enter crop name (leave blank for all crops): ")
            if crop_name.strip() == "":
                crop_name = None
                
            result = self.predict_crop_prices(crop_name, historical_data_path)
            
            if isinstance(result, dict):
                print("\nPredicted prices for the next 5 days:")
                for day in result['prediction']:
                    if 'date' in day:
                        print(f"Date: {day['date']}, Price: ${day['price']:.2f}")
                    else:
                        print(f"Day {day['day']}: ${day['price']:.2f}")
                        
                print(f"\nA visualization has been saved as 'price_prediction_{crop_name if crop_name else 'all_crops'}.png'")
            else:
                print(result)  # Error message
        except Exception as e:
            print(f"Error: {e}")


# Main entry point
if __name__ == "__main__":
    assistant = AgriculturalAssistant()
    assistant.run_interactive()