#crop_reccomendation
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('crop_recommendation/crop_recommendation.csv')

# Feature and target separation
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a simpler model for quick setup
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model components
joblib.dump(rf_model, 'best_rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
print('Crop recommendation model trained and saved!')














#fertilzier_reccomendarion
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
fertilizer_data = pd.read_csv('fertilizer_recommendation/fertilizer.csv')

# Encode categorical columns
label_encoders = {}
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    le = LabelEncoder()
    fertilizer_data[col] = le.fit_transform(fertilizer_data[col])
    label_encoders[col] = le

# Save label encoders
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

# Define features and target
X = fertilizer_data.drop(columns=['Fertilizer Name'])
y = fertilizer_data['Fertilizer Name']

# Train the model
fertilizer_model = RandomForestClassifier(random_state=42)
fertilizer_model.fit(X, y)

# Save the trained model
with open('fertilizer_model.pkl', 'wb') as model_file:
    pickle.dump(fertilizer_model, model_file)
    
print('Fertilizer recommendation model trained and saved!')

