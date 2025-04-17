# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from main import AgriculturalAssistant
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Agricultural Assistant
assistant = AgriculturalAssistant(base_path=os.path.dirname(os.path.abspath(__file__)))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        try:
            # Get form data
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorous'])
            K = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Get recommendation
            crop = assistant.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
            
            if isinstance(crop, str) and "Error" in crop:
                flash(crop, 'error')
                return redirect(url_for('crop_recommendation'))
            
            # Store result in session to display on results page
            result = {
                'input': {
                    'N': N,
                    'P': P,
                    'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                },
                'recommendation': crop
            }
            
            return render_template('results/crop_result.html', result=result)
            
        except ValueError as e:
            flash('Please enter valid numeric values for all fields.', 'error')
            return redirect(url_for('crop_recommendation'))
    
    return render_template('crop_recommend.html')

@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == 'POST':
        try:
            # Get form data
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            moisture = float(request.form['moisture'])
            soil_type = request.form['soil_type']
            crop_type = request.form['crop_type']
            nitrogen = float(request.form['nitrogen'])
            potassium = float(request.form['potassium'])
            phosphorous = float(request.form['phosphorous'])
            
            # Get recommendation
            result = assistant.recommend_fertilizer(
                temperature, humidity, moisture, soil_type, crop_type, 
                nitrogen, potassium, phosphorous
            )
            
            if isinstance(result, str):
                flash(result, 'error')
                return redirect(url_for('fertilizer_recommendation'))
            
            # Prepare result for template
            display_result = {
                'input': {
                    'temperature': temperature,
                    'humidity': humidity,
                    'moisture': moisture,
                    'soil_type': soil_type,
                    'crop_type': crop_type,
                    'nitrogen': nitrogen,
                    'potassium': potassium,
                    'phosphorous': phosphorous
                },
                'fertilizer': result['fertilizer'],
                'suggestions': result['suggestions']
            }
            
            return render_template('results/fertilizer_result.html', result=display_result)
            
        except ValueError as e:
            flash('Please enter valid values for all fields.', 'error')
            return redirect(url_for('fertilizer_recommendation'))
    
    # Get available soil and crop types for dropdowns
    soil_types = []
    crop_types = []
    if assistant.fertilizer_encoders:
        if 'Soil Type' in assistant.fertilizer_encoders:
            soil_types = list(assistant.fertilizer_encoders['Soil Type'].classes_)
        if 'Crop Type' in assistant.fertilizer_encoders:
            crop_types = list(assistant.fertilizer_encoders['Crop Type'].classes_)
    
    return render_template('fertilizer_recommend.html', 
                         soil_types=soil_types, 
                         crop_types=crop_types)

@app.route('/price_prediction', methods=['GET', 'POST'])
def price_prediction():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'price_data' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('price_prediction'))
            
        file = request.files['price_data']
        
        # If user does not select file, browser might submit empty file
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('price_prediction'))
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            crop_name = request.form['crop_name'].strip() or None
            
            # Get prediction
            result = assistant.predict_crop_prices(crop_name, filepath)
            
            if isinstance(result, str):
                flash(result, 'error')
                return redirect(url_for('price_prediction'))
            
            # Prepare result for template
            display_result = {
                'crop_name': crop_name or 'All Crops',
                'predictions': result['prediction'],
                'plot_file': result['plot_file'].replace('\\', '/')  # Fix path for web
            }
            
            return render_template('results/price_result.html', result=display_result)
    
    return render_template('price_predict.html')

@app.route('/generate_report', methods=['GET', 'POST'])
def generate_report():
    if request.method == 'POST':
        try:
            # Get form data
            crop_name = request.form['crop_name']
            
            soil_data = {
                'nitrogen': float(request.form['nitrogen']),
                'phosphorous': float(request.form['phosphorous']),
                'potassium': float(request.form['potassium']),
                'ph': float(request.form['ph']),
                'soil_type': request.form['soil_type']
            }
            
            weather_data = {
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'rainfall': float(request.form['rainfall']),
                'moisture': float(request.form['moisture'])
            }
            
            # Handle price data file
            price_data_path = None
            if 'price_data' in request.files:
                file = request.files['price_data']
                if file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    price_data_path = filepath
            
            # Generate report
            report = assistant.generate_report(
                crop_name, 
                soil_data, 
                weather_data, 
                price_data_path
            )
            
            if isinstance(report, str):
                flash(report, 'error')
                return redirect(url_for('generate_report'))
            
            # Export report
            export_format = request.form.get('export_format', 'text')
            export_result = assistant.export_report(report, export_format)
            
            if export_result.startswith("Error"):
                flash(export_result, 'error')
                return redirect(url_for('generate_report'))
            
            # Prepare result for template
            display_result = {
                'report': report,
                'export_message': export_result,
                'export_format': export_format
            }
            
            return render_template('report.html', result=display_result)
            
        except ValueError as e:
            flash(f'Please enter valid values: {str(e)}', 'error')
            return redirect(url_for('generate_report'))
    
    # Get available soil types for dropdown
    soil_types = []
    if assistant.fertilizer_encoders and 'Soil Type' in assistant.fertilizer_encoders:
        soil_types = list(assistant.fertilizer_encoders['Soil Type'].classes_)
    
    return render_template('report_form.html', soil_types=soil_types)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory('outputs', filename)

@app.route('/data/<filename>')
def data_file(filename):
    return send_from_directory('data', filename)

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    app.run(debug=True)