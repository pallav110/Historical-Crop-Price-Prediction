import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory , session
from werkzeug.utils import secure_filename
from main import AgriculturalAssistant
from supabase import create_client, Client
from dotenv import load_dotenv
from functools import wraps

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Agricultural Assistant
assistant = AgriculturalAssistant(base_path=os.path.dirname(os.path.abspath(__file__)))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'token' not in session:
            flash("You must be logged in to access that page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        res = supabase.auth.sign_up({"email": email, "password": password})

        if res.user:
            flash("Sign-up successful! Please check your email.")
            return redirect(url_for('login'))
        elif res.error:
            flash(res.error.message)
        else:
            flash("Sign-up failed.")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if res.session:
                session['token'] = res.session.access_token
                flash('Logged in successfully!')
                return redirect(url_for('index'))
            else:
                flash('Login failed')
        except Exception as e:
            if hasattr(e, 'message') and 'Email not confirmed' in str(e):
                flash('Email not confirmed. Please check your inbox.')
            else:
                flash(f'Login failed: {e}')
    return render_template('login.html')

@app.route('/index')
def index():
    if 'token' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('token', None)
    flash('Logged out successfully.')
    return redirect(url_for('login'))

@app.route('/crop_recommendation', methods=['GET', 'POST'])
@login_required
def crop_recommendation():
    if request.method == 'POST':
        try:
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorous'])
            K = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            crop = assistant.recommend_crop(N, P, K, temperature, humidity, ph, rainfall)

            if isinstance(crop, str) and "Error" in crop:
                flash(crop, 'error')
                return redirect(url_for('crop_recommendation'))

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

        except ValueError:
            flash('Please enter valid numeric values for all fields.', 'error')
            return redirect(url_for('crop_recommendation'))

    return render_template('crop_recommend.html')

@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
@login_required
def fertilizer_recommendation():
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            moisture = float(request.form['moisture'])
            soil_type = request.form['soil_type']
            crop_type = request.form['crop_type']
            nitrogen = float(request.form['nitrogen'])
            potassium = float(request.form['potassium'])
            phosphorous = float(request.form['phosphorous'])

            result = assistant.recommend_fertilizer(
                temperature, humidity, moisture, soil_type, crop_type,
                nitrogen, potassium, phosphorous
            )

            if isinstance(result, str):
                flash(result, 'error')
                return redirect(url_for('fertilizer_recommendation'))

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

        except ValueError:
            flash('Please enter valid values for all fields.', 'error')
            return redirect(url_for('fertilizer_recommendation'))

    # Safe access to encoders
    soil_types = []
    crop_types = []
    if assistant.fertilizer_encoders:
        if 'Soil Type' in assistant.fertilizer_encoders:
            encoder = assistant.fertilizer_encoders['Soil Type']
            if hasattr(encoder, 'classes_'):
                soil_types = list(encoder.classes_)
        if 'Crop Type' in assistant.fertilizer_encoders:
            encoder = assistant.fertilizer_encoders['Crop Type']
            if hasattr(encoder, 'classes_'):
                crop_types = list(encoder.classes_)

    return render_template('fertilizer_recommend.html',
                           soil_types=soil_types,
                           crop_types=crop_types)

@app.route('/price_prediction', methods=['GET', 'POST'])
@login_required
def price_prediction():
    if request.method == 'POST':
        crop_name = request.form['crop_name'].strip()

        # Fixed path to the CSV file
        fixed_csv_path = 'crop_price_prediction/historical_prices.csv'

        result = assistant.predict_crop_prices(crop_name, fixed_csv_path)

        if isinstance(result, str):
            flash(result, 'error')
            return redirect(url_for('price_prediction'))

        display_result = {
            'crop_name': crop_name,
            'predictions': result['prediction'],
            'plot_file': result['plot_file'].replace('\\', '/')  # Ensure web-safe path
        }

        return render_template('results/price_result.html', result=display_result)

    return render_template('price_predict.html')

@app.route('/generate_report', methods=['GET', 'POST'])
@login_required
def generate_report():
    if request.method == 'POST':
        try:
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

            # Optional: use fixed price file path
            price_data_path = 'crop_price_prediction/historical_prices.csv'

            report = assistant.generate_report(
                crop_name,
                soil_data,
                weather_data,
                price_data_path
            )

            if isinstance(report, str):
                flash(report, 'error')
                return redirect(url_for('generate_report'))

            export_format = request.form.get('export_format', 'text')
            export_result = assistant.export_report(report, export_format)

            if export_result.startswith("Error"):
                flash(export_result, 'error')
                return redirect(url_for('generate_report'))

            display_result = {
                'report': report,
                'export_message': export_result,
                'export_format': export_format
            }

            return render_template('report.html', result=display_result)

        except ValueError as e:
            flash(f'Please enter valid values: {str(e)}', 'error')
            return redirect(url_for('generate_report'))

    # Safe check for encoder and class-based encodings
    soil_types = []
    if hasattr(assistant, 'fertilizer_encoders'):
        encoder = assistant.fertilizer_encoders.get('Soil Type')
        if hasattr(encoder, 'classes_'):
            soil_types = list(encoder.classes_)

    return render_template('report_form.html', soil_types=soil_types)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory('outputs', filename)

@app.route('/data/<filename>')
def data_file(filename):
    return send_from_directory('data', filename)

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    app.run(debug=True, port=7000)
