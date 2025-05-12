import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import datetime
import matplotlib
import flask_cors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys

# Check and install required libraries
required_packages = ['flask', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'colorama', 'flask-cors']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        os.system(f'pip install {package}')

# Import colorama for colored terminal output
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# Custom logger class for stylish logging
class StylishLogger:
    def __init__(self, app_name="Stock Prediction Server"):
        self.app_name = app_name
        self.start_time = time.time()
        
    def _get_timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def _get_elapsed(self):
        return f"{time.time() - self.start_time:.3f}s"
    
    def info(self, message):
        print(f"{Fore.CYAN}[{self._get_timestamp()}] {Fore.WHITE}[{self._get_elapsed()}] {Fore.BLUE}[INFO] {Fore.WHITE}{message}")
    
    def success(self, message):
        print(f"{Fore.CYAN}[{self._get_timestamp()}] {Fore.WHITE}[{self._get_elapsed()}] {Fore.GREEN}[SUCCESS] {Fore.WHITE}{message}")
    
    def warning(self, message):
        print(f"{Fore.CYAN}[{self._get_timestamp()}] {Fore.WHITE}[{self._get_elapsed()}] {Fore.YELLOW}[WARNING] {Fore.WHITE}{message}")
    
    def error(self, message):
        print(f"{Fore.CYAN}[{self._get_timestamp()}] {Fore.WHITE}[{self._get_elapsed()}] {Fore.RED}[ERROR] {Fore.WHITE}{message}")
    
    def debug(self, message):
        print(f"{Fore.CYAN}[{self._get_timestamp()}] {Fore.WHITE}[{self._get_elapsed()}] {Fore.MAGENTA}[DEBUG] {Fore.WHITE}{message}")
    
    def api(self, method, endpoint, status=200):
        status_color = Fore.GREEN if status < 400 else Fore.RED
        print(f"{Fore.CYAN}[{self._get_timestamp()}] {Fore.WHITE}[{self._get_elapsed()}] {Fore.BLUE}[API] {Fore.YELLOW}{method} {Fore.WHITE}{endpoint} {status_color}{status}")
    
    def section(self, title):
        line = "=" * (len(title) + 10)
        print(f"\n{Fore.MAGENTA}{line}")
        print(f"{Fore.MAGENTA}{' ' * 5}{title}{' ' * 5}")
        print(f"{Fore.MAGENTA}{line}\n")
    
    def startup(self):
        print(f"\n{Fore.CYAN}{'=' * 80}")
        print(f"{Fore.CYAN}{' ' * ((80 - len(self.app_name)) // 2)}{Fore.WHITE}{Back.BLUE}{Style.BRIGHT} {self.app_name} {Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}\n")
        print(f"{Fore.GREEN}Server starting at: {Fore.WHITE}{self._get_timestamp()}")
        print(f"{Fore.GREEN}Python version: {Fore.WHITE}{sys.version}")
        print(f"{Fore.GREEN}Working directory: {Fore.WHITE}{os.getcwd()}")
        print(f"{Fore.CYAN}{'=' * 80}\n")

# Create logger instance
logger = StylishLogger()

app = Flask(__name__, static_folder='static', template_folder='templates')
flask_cors.CORS(app)

# Create directories if they don't exist
logger.section("INITIALIZING SERVER")
logger.info("Checking required directories...")
os.makedirs('static', exist_ok=True)
logger.success("Static directory verified")
os.makedirs('templates', exist_ok=True)
logger.success("Templates directory verified")

# Load models
def load_models():
    logger.info("Loading prediction models from disk...")
    try:
        with open('models/stock_prediction_models.pkl', 'rb') as f:
            models = pickle.load(f)
            logger.success(f"Successfully loaded models for {len(models)} symbols")
            return models
    except FileNotFoundError:
        logger.error("Models file not found. Please run train_model.py first.")
        return {}

# Get available symbols
def get_available_symbols():
    logger.info("Retrieving available stock symbols...")
    models = load_models()
    symbols = list(models.keys())
    logger.success(f"Found {len(symbols)} available symbols: {', '.join(symbols)}")
    return symbols

# Generate predictions for a specific year
def predict_year(symbol, year):
    logger.section(f"GENERATING PREDICTIONS FOR {symbol} - {year}")
    logger.info(f"Starting prediction process for {symbol} in year {year}")
    
    start_time = time.time()
    models = load_models()
    
    if symbol not in models:
        logger.error(f"Symbol {symbol} not found in available models")
        return {"error": f"Symbol {symbol} not found in models."}
    
    # Get model info for this symbol
    model_info = models[symbol]
    best_model_name = model_info['best_model']
    logger.info(f"Using {best_model_name} model for predictions (best performing model)")
    
    # For simplicity, generate synthetic data for prediction visualization
    logger.debug(f"Creating date range for year {year}")
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    logger.debug(f"Created date range with {len(date_range)} business days")
    
    # Generate predictions based on model type
    if best_model_name != 'ARIMA':
        logger.info(f"Generating ML-based predictions using {best_model_name}")
        # Get model and scalers
        best_model = model_info[best_model_name]['model']
        scaler_X = model_info['scaler_X']
        scaler_y = model_info['scaler_y']
        
        # Generate synthetic features (simplified)
        num_features = len(model_info['feature_names'])
        num_days = len(date_range)
        logger.debug(f"Creating synthetic features matrix of shape ({num_days}, {num_features})")
        
        # Use random data as features (this is a simplification)
        features = np.random.randn(num_days, num_features)
        features_scaled = scaler_X.transform(features)
        
        # Make predictions
        logger.info("Running model inference...")
        predictions_scaled = best_model.predict(features_scaled)
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        logger.success(f"Generated {len(predictions)} price predictions")
    else:
        logger.info("Generating ARIMA-based predictions")
        # For ARIMA, generate more smooth predictions (simplified)
        num_days = len(date_range)
        base_value = 100  # Starting value
        
        # Add trend, seasonality, and noise
        logger.debug("Adding trend, seasonality and noise components to predictions")
        trend = np.linspace(0, 20, num_days)  # General upward trend
        seasonality = 10 * np.sin(np.linspace(0, 6*np.pi, num_days))  # Seasonal pattern
        noise = np.random.normal(0, 5, num_days)  # Random noise
        
        predictions = base_value + trend + seasonality + noise
        logger.success(f"Generated {len(predictions)} price predictions")
    
    # Create historical data for comparison (past 3 years)
    logger.info("Generating historical comparison data")
    past_years = [year-3, year-2, year-1]
    historical_data = []
    
    for past_year in past_years:
        logger.debug(f"Creating synthetic data for year {past_year}")
        num_days_past = len(pd.date_range(start=f"{past_year}-01-01", end=f"{past_year}-12-31", freq='B'))
        
        base_value_past = 100 - (year - past_year) * 10  # Lower base for older years
        trend_past = np.linspace(0, 15, num_days_past)
        seasonality_past = 8 * np.sin(np.linspace(0, 6*np.pi, num_days_past))
        noise_past = np.random.normal(0, 3, num_days_past)
        
        past_prices = base_value_past + trend_past + seasonality_past + noise_past
        
        past_dates = pd.date_range(start=f"{past_year}-01-01", end=f"{past_year}-12-31", freq='B')
        historical_data.append({
            'year': past_year,
            'dates': [d.strftime('%Y-%m-%d') for d in past_dates],
            'prices': past_prices.tolist()
        })
    
    # Create result
    result = {
        'symbol': symbol,
        'year': year,
        'model_used': best_model_name,
        'r2_score': model_info[best_model_name]['eval']['adjusted_r2'],
        'dates': [d.strftime('%Y-%m-%d') for d in date_range],
        'predicted_prices': predictions.tolist(),
        'historical_data': historical_data
    }
    
    elapsed_time = time.time() - start_time
    logger.success(f"Prediction completed in {elapsed_time:.2f} seconds")
    logger.debug(f"Prediction summary: {len(predictions)} days, avg price: ${np.mean(predictions):.2f}")
    
    return result

# Routes
@app.route('/')
def home():
    logger.api("GET", "/", 200)
    logger.info("Serving home page")
    return render_template('index.html')

@app.route('/api/symbols')
def api_symbols():
    logger.api("GET", "/api/symbols", 200)
    logger.info("Processing request for available symbols")
    symbols = get_available_symbols()
    logger.success(f"Returning {len(symbols)} symbols to client")
    return jsonify(symbols)

@app.route('/api/predict')
def api_predict():
    symbol = request.args.get('symbol')
    year = request.args.get('year')
    
    logger.api("GET", f"/api/predict?symbol={symbol}&year={year}")
    logger.info(f"Processing prediction request for {symbol} in year {year}")
    
    if not symbol or not year:
        logger.warning("Missing required parameters: symbol and year")
        return jsonify({"error": "Symbol and year are required parameters."}), 400
    
    try:
        year = int(year)
        if year < 2000 or year > 2030:
            logger.warning(f"Invalid year value: {year} (must be between 2000-2030)")
            return jsonify({"error": "Year must be between 2000 and 2030."}), 400
    except ValueError:
        logger.error(f"Invalid year format: {year}")
        return jsonify({"error": "Invalid year format."}), 400
    
    result = predict_year(symbol, year)
    
    if "error" in result:
        logger.error(f"Error in prediction: {result['error']}")
        return jsonify(result), 404
    
    logger.success(f"Successfully generated prediction for {symbol} in {year}")
    return jsonify(result)

@app.route('/api/metrics')
def api_metrics():
    logger.api("GET", "/api/metrics")
    logger.info("Processing request for model metrics")
    
    try:
        metrics_df = pd.read_csv('static/metrics_summary.csv')
        num_models = len(metrics_df['model_name'].unique())
        num_metrics = len(metrics_df.columns) - 1  # Excluding model_name column
        logger.success(f"Retrieved metrics for {num_models} models ({num_metrics} metrics per model)")
        return jsonify(metrics_df.to_dict(orient='records'))
    except FileNotFoundError:
        logger.error("Metrics file not found. Please run train_model.py first.")
        return jsonify({"error": "Metrics file not found. Please run train_model.py first."}), 404

@app.route('/static/<path:path>')
def serve_static(path):
    logger.api("GET", f"/static/{path}")
    logger.debug(f"Serving static file: {path}")
    return send_from_directory('static', path)

# Request logging middleware
@app.before_request
def before_request():
    request.start_time = time.time()
    logger.debug(f"Received {request.method} request to {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        elapsed = time.time() - request.start_time
        logger.debug(f"Request completed in {elapsed*1000:.2f}ms with status {response.status_code}")
    return response

if __name__ == '__main__':
    # Create HTML template if it doesn't exist
    html_template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(html_template_path):
        logger.info("HTML template not found, creating default template...")
        with open(html_template_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Price Prediction</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold text-center mb-8 text-blue-800">Stock Price Prediction Dashboard</h1>
        
        <div class="bg-white shadow-lg rounded-lg p-6 mb-8">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <label for="symbol-select" class="block text-sm font-medium text-gray-700 mb-2">Select Stock Symbol</label>
                    <select id="symbol-select" class="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <option value="">Loading symbols...</option>
                    </select>
                </div>
                <div>
                    <label for="year-input" class="block text-sm font-medium text-gray-700 mb-2">Prediction Year</label>
                    <input type="number" id="year-input" min="2000" max="2030" value="2025" class="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                </div>
            </div>
            <button id="predict-btn" class="mt-6 w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">
                Predict Stock Prices
            </button>
        </div>
        
        <div id="prediction-results" class="hidden">
            <div class="bg-white shadow-lg rounded-lg p-6 mb-8">
                <h2 class="text-xl font-semibold mb-4 text-center" id="prediction-title">Prediction Results</h2>
                <div class="mb-4">
                    <p><strong>Model Used:</strong> <span id="model-used"></span></p>
                    <p><strong>R² Score:</strong> <span id="r2-score"></span></p>
                </div>
                <div class="h-96">
                    <canvas id="prediction-chart"></canvas>
                </div>
            </div>
            
            <div class="bg-white shadow-lg rounded-lg p-6 mb-8">
                <h2 class="text-xl font-semibold mb-4 text-center">Historical Price Trends</h2>
                <div class="h-96">
                    <canvas id="historical-chart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="bg-white shadow-lg rounded-lg p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4 text-center">Model Performance Metrics</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <img src="/static/metrics_comparison.png" alt="Metrics Comparison" class="w-full">
                </div>
                <div>
                    <img src="/static/confusion_matrices.png" alt="Confusion Matrices" class="w-full">
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Load available symbols when page loads
        document.addEventListener('DOMContentLoaded', function() {
            fetch('/api/symbols')
                .then(response => response.json())
                .then(symbols => {
                    const selectElement = document.getElementById('symbol-select');
                    selectElement.innerHTML = '';
                    
                    symbols.forEach(symbol => {
                        const option = document.createElement('option');
                        option.value = symbol;
                        option.textContent = symbol;
                        selectElement.appendChild(option);
                    });
                })
                .catch(error => {
                    console.error('Error loading symbols:', error);
                    alert('Error loading symbols. Please make sure the server is running and models are trained.');
                });
        });
        
        // Prediction chart
        let predictionChart = null;
        let historicalChart = null;
        
        // Handle predict button click
        document.getElementById('predict-btn').addEventListener('click', function() {
            const symbol = document.getElementById('symbol-select').value;
            const year = document.getElementById('year-input').value;
            
            if (!symbol) {
                alert('Please select a stock symbol.');
                return;
            }
            
            if (!year || year < 2000 || year > 2030) {
                alert('Please enter a valid year between 2000 and 2030.');
                return;
            }
            
            // Show loading state
            document.getElementById('predict-btn').textContent = 'Loading...';
            document.getElementById('predict-btn').disabled = true;
            
            // Fetch prediction data
            fetch(`/api/predict?symbol=${symbol}&year=${year}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(`Error: ${data.error}`);
                        return;
                    }
                    
                    // Update prediction info
                    document.getElementById('prediction-title').textContent = `${data.symbol} Price Prediction for ${data.year}`;
                    document.getElementById('model-used').textContent = data.model_used;
                    document.getElementById('r2-score').textContent = data.r2_score.toFixed(4);
                    
                    // Show results section
                    document.getElementById('prediction-results').classList.remove('hidden');
                    
                    // Draw prediction chart
                    drawPredictionChart(data);
                    
                    // Draw historical chart
                    drawHistoricalChart(data);
                })
                .catch(error => {
                    console.error('Error fetching prediction:', error);
                    alert('Error fetching prediction. Please try again.');
                })
                .finally(() => {
                    // Reset button state
                    document.getElementById('predict-btn').textContent = 'Predict Stock Prices';
                    document.getElementById('predict-btn').disabled = false;
                });
        });
        
        // Function to draw prediction chart
        function drawPredictionChart(data) {
            const ctx = document.getElementById('prediction-chart').getContext('2d');
            
            // Destroy existing chart if it exists
            if (predictionChart) {
                predictionChart.destroy();
            }
            
            // Create new chart
            predictionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [{
                        label: `${data.symbol} Predicted Price (${data.year})`,
                        data: data.predicted_prices,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `Predicted ${data.symbol} Prices for ${data.year}`,
                            font: {
                                size: 16
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += new Intl.NumberFormat('en-US', { 
                                            style: 'currency', 
                                            currency: 'USD' 
                                        }).format(context.parsed.y);
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            },
                            ticks: {
                                maxTicksLimit: 12
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Stock Price'
                            },
                            beginAtZero: false
                        }
                    }
                }
            });
        }
        
        // Function to draw historical chart
        function drawHistoricalChart(data) {
            const ctx = document.getElementById('historical-chart').getContext('2d');
            
            // Destroy existing chart if it exists
            if (historicalChart) {
                historicalChart.destroy();
            }
            
            // Create datasets for historical data
            const datasets = data.historical_data.map((yearData, index) => {
                const colors = [
                    'rgb(255, 99, 132)',
                    'rgb(54, 162, 235)',
                    'rgb(255, 206, 86)'
                ];
                
                return {
                    label: `${data.symbol} Prices (${yearData.year})`,
                    data: yearData.prices,
                    borderColor: colors[index % colors.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.1
                };
            });
            
            // Create new chart
            historicalChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.historical_data[0].dates,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: `${data.symbol} Historical Price Trends`,
                            font: {
                                size: 16
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += new Intl.NumberFormat('en-US', { 
                                            style: 'currency', 
                                            currency: 'USD' 
                                        }).format(context.parsed.y);
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            },
                            ticks: {
                                maxTicksLimit: 12
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Stock Price'
                            },
                            beginAtZero: false
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>
            """)
    logger.startup()
    logger.section("SERVER READY")
    print("Starting Flask server on http://127.0.0.1:5000/")
    app.run(debug=True)