# Agricultural Assistant

An integrated system for crop recommendation, fertilizer recommendation, and crop price prediction.

## Project Overview

This project combines three key agricultural assistance tools:

1. **Crop Recommendation System**: Suggests suitable crops based on soil parameters and environmental conditions.
2. **Fertilizer Recommendation System**: Recommends appropriate fertilizers based on soil nutrients and crop type.
3. **Crop Price Prediction System**: Forecasts crop prices for the next 5 days based on historical price data.

## System Architecture

### 1. Crop Recommendation System
- Uses a Random Forest Classifier to recommend crops based on soil composition and environmental factors.
- Input features: N (Nitrogen), P (Phosphorus), K (Potassium), temperature, humidity, pH, and rainfall.
- Output: Recommended crop type.

### 2. Fertilizer Recommendation System
- Recommends fertilizers based on soil nutrient levels and crop requirements.
- Input features: Temperature, humidity, moisture, soil type, crop type, nitrogen, potassium, and phosphorous values.
- Output: Recommended fertilizer and nutrient improvement suggestions.

### 3. Crop Price Prediction System
- Uses a LSTM (Long Short-Term Memory) neural network to predict future crop prices.
- Input: Historical price data for specific crops.
- Output: Predicted prices for the next 5 days and a visualization graph.

## Project Structure
## Follow this structure only thoroughly.

```
agricultural-assistant/
│
├── crop_recommendation/
│   ├── best_rf_model.joblib    # Trained Random Forest model
│   ├── scaler.joblib           # Feature scaler
│   ├── label_encoder.joblib    # Label encoder for crop types
│   └── crop_recommendation.csv # Training data
│
├── fertilizer_recommendation/
│   ├── fertilizer_model.pkl    # Trained fertilizer recommendation model
│   ├── label_encoders.pkl      # Label encoders for categorical features
│   └── fertilizer.csv          # Training data
│
├── crop_price_prediction/
│   ├── lstm_model.h5           # Trained LSTM model
│   ├── price_scaler.pkl        # Price scaler
│   └── historical_prices.csv   # Historical price data
│
├──static/styles.css
├──templates/
│     ├──reports/
│     │    ├──crop_results.html
│     │    ├──fertilizer_results.html
│     │    └──price_results.html
│     ├──base.html
│     ├──crop_recommend.html
│     ├──index.html
│     ├──fertilizer.html
│     ├──price_predict.html
│     ├──report_form.html
│     └──report.html
├──app.py                      #Flask implementation
├── main.py                     # Main application entry point
├── crop_price_predictor.py     # Standalone price prediction module
├──README.md                   # This file
└── setup.txt

```

## Data Requirements

### Crop Recommendation Data
Format:
```
N,P,K,temperature,humidity,ph,rainfall,label
90,42,43,20.87,82.00,6.50,202.93,rice
85,58,41,21.77,80.31,7.03,226.65,rice
...
```

### Fertilizer Recommendation Data
Format:
```
Temperature,Humidity,Moisture,Soil Type,Crop Type,Nitrogen,Potassium,Phosphorous,Fertilizer Name
26,52,38,Sandy,Maize,37,0,0,Urea
29,52,45,Loamy,Sugarcane,12,0,36,DAP
...
```

### Crop Price Historical Data
Format:
```
date,crop_name,price,market_location
2023-01-01,rice,24.50,Mumbai
2023-01-02,rice,24.75,Mumbai
...
```

## How to Use

### Setup
1. Install required dependencies:
   ```
   pip install pandas numpy scikit-learn tensorflow joblib matplotlib
   ```

2. Place your CSV data files in their respective directories.

### Running the Application
Execute the scripts below and train models on your own , if needed tweaking , you can do so:
```
python crop_price_predictor.py
python crop_and_fertilizer_reccomend.py
```
For price prediction the models would be saved in crop_price_model after running the code
Move it from that folder to crop_price_prediction.
after the models are saved update the path if necessary else run:
```
python main.py
```

This will start the interactive menu where you can choose different functions:
1. Crop Recommendation
2. Fertilizer Recommendation
3. Crop Price Prediction
4. Exit

## Technical Implementation

### Crop Recommendation Model
- Uses Random Forest Classifier to handle the multi-class classification task.
- Features are standardized using StandardScaler.
- Uses GridSearchCV for hyperparameter tuning.

### Fertilizer Recommendation Model
- Uses Random Forest Classifier for recommending fertilizers.
- Incorporates domain knowledge through a dictionary of nutrient-specific suggestions.

### Price Prediction Model
- Uses LSTM neural network architecture which is well-suited for time series forecasting.
- Sequence length of 30 days (uses past month's data to predict next 5 days).
- Preprocessing includes scaling the prices and creating sequential data.

## Extending the Project

### Adding More Crops/Fertilizers
- Update the respective CSV files with new data.
- Retrain the models using the original scripts.

### Improving Price Prediction
- Consider adding more features such as weather data, seasonal trends, or economic indicators.
- Experiment with different model architectures like GRU, bidirectional LSTM, or transformer models.

### Web Interface
- The system can be extended with a web interface using frameworks like Flask or Django.
- APIs can be created to expose the functionality to other applications.

## Limitations and Future Work

- The current crop recommendation doesn't account for seasonal variations.
- The fertilizer recommendation doesn't consider economic factors or availability.
- Price prediction could be improved with more robust data and additional features.
- Integration with real-time weather data and market information would enhance predictions.

## Conclusion

This Agricultural Assistant provides a comprehensive solution for farmers and agricultural professionals, combining soil analysis, fertilizer recommendation, and price prediction in one integrated system. By utilizing machine learning and deep learning techniques, it offers data-driven insights to optimize agricultural practices and decision-making.