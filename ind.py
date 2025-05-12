from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
# Make lightgbm optional
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available, will skip this model")

from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Input
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models and data
datasets = {}
scaler = MinMaxScaler(feature_range=(0, 1))
model_metrics = {}

def load_dataset(file_path):
    """Load and preprocess dataset"""
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    logger.info(f"Dataset loaded successfully with shape {df.shape}")
    return df

def preprocess_data(df, symbol, feature_columns, target_column='Close', seq_length=60):
    """Preprocess data for ML models"""
    logger.info(f"Preprocessing data for symbol {symbol} with features {feature_columns}")
    
    # Filter data for specific symbol if multiple symbols exist
    if 'Symbol' in df.columns:
        symbol_data = df[df['Symbol'] == symbol]
        logger.info(f"Filtered data for symbol {symbol} with shape {symbol_data.shape}")
    else:
        symbol_data = df
        logger.info(f"Using entire dataset with shape {symbol_data.shape}")
    
    # Select features and target
    data = symbol_data[feature_columns + [target_column]].values
    logger.info(f"Selected features and target with shape {data.shape}")
    
    # Scale the data
    scaled_data = scaler.fit_transform(data)
    logger.info("Data scaled successfully")
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, :])
        y.append(scaled_data[i, feature_columns.index(target_column) if target_column in feature_columns else -1])
    
    X, y = np.array(X), np.array(y)
    logger.info(f"Created sequences with shapes X: {X.shape}, y: {y.shape}")
    
    return X, y

# Model definitions
def build_lstm_model(input_shape):
    logger.info(f"Building LSTM model with input shape {input_shape}")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info("LSTM model built successfully")
    return model

def build_hatr_model(input_shape):
    # Simplified HATR implementation
    inputs = Input(shape=input_shape)
    lstm1 = LSTM(50, return_sequences=True)(inputs)
    # Use manual attention mechanism instead of the Attention layer
    # which might have compatibility issues
    attention_scores = Dense(1, activation='tanh')(lstm1)
    attention_weights = tf.nn.softmax(attention_scores, axis=1)
    context_vector = tf.reduce_sum(lstm1 * attention_weights, axis=1)
    
    lstm2 = Dense(50)(context_vector)
    outputs = Dense(1)(lstm2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm = LSTM(50, return_sequences=True)(inputs)
    
    # Manual attention mechanism
    attention_scores = Dense(1, activation='tanh')(lstm)
    attention_weights = tf.nn.softmax(attention_scores, axis=1)
    context_vector = tf.reduce_sum(lstm * attention_weights, axis=1)
    
    flatten = Flatten()(lstm)  # Fallback if context vector doesn't work
    combined = tf.concat([context_vector, flatten], axis=-1)
    output = Dense(1)(combined)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_lstm_xgboost_model(X_train, y_train):
    # First train LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = build_lstm_model(input_shape)
    lstm_model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
    
    # Get LSTM predictions as features for XGBoost
    lstm_features = lstm_model.predict(X_train)
    
    # Reshape X_train for XGBoost
    xgb_features = X_train.reshape(X_train.shape[0], -1)
    xgb_features = np.hstack((xgb_features, lstm_features))
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb_model.fit(xgb_features, y_train)
    
    return {"lstm": lstm_model, "xgboost": xgb_model}

def train_model(model_type, X_train, y_train, X_test, y_test):
    """Train the selected model and evaluate metrics"""
    logger.info(f"Training model: {model_type}")
    
    if model_type == "LSTM":
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_data=(X_test, y_test))
        
        # Get predictions for metrics
        y_pred = model.predict(X_test, verbose=0)
        
        # Log training metrics
        logger.info(f"LSTM training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"LSTM validation loss: {history.history['val_loss'][-1]:.4f}")
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        model_metrics[model_type] = metrics
        
        return model
    
    elif model_type == "HATR":
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_hatr_model(input_shape)
        history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_data=(X_test, y_test))
        
        # Get predictions for metrics
        y_pred = model.predict(X_test, verbose=0)
        
        # Log training metrics
        logger.info(f"HATR training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"HATR validation loss: {history.history['val_loss'][-1]:.4f}")
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        model_metrics[model_type] = metrics
        
        return model
    
    elif model_type == "XGBoost":
        # Reshape data for XGBoost
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        logger.info(f"XGBoost input shape: {X_train_reshaped.shape}")
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train_reshaped, y_train)
        
        # Get predictions for metrics
        y_pred = model.predict(X_test_reshaped)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        model_metrics[model_type] = metrics
        
        logger.info(f"XGBoost training completed with metrics: {metrics}")
        return model
    
    elif model_type == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM is not available, please install it or select another model")
            raise ValueError("LightGBM is not available")
            
        # Reshape data for LightGBM
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        logger.info(f"LightGBM input shape: {X_train_reshaped.shape}")
        model = lgb.LGBMRegressor(n_estimators=100)
        model.fit(X_train_reshaped, y_train)
        
        # Get predictions for metrics
        y_pred = model.predict(X_test_reshaped)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        model_metrics[model_type] = metrics
        
        logger.info(f"LightGBM training completed with metrics: {metrics}")
        return model
    
    elif model_type == "ARIMA":
        # Simplified ARIMA implementation
        try:
            model = ARIMA(y_train, order=(5,1,0))
            model_fit = model.fit()
            return model_fit
        except Exception as e:
            logger.error(f"ARIMA model failed: {str(e)}")
            logger.info("Falling back to simpler ARIMA model")
            # Try with simpler parameters
            model = ARIMA(y_train, order=(1,0,0))
            model_fit = model.fit()
            return model_fit
    
    elif model_type == "SVM":
        # Reshape data for SVM
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        model = SVR(kernel='rbf')
        model.fit(X_reshaped, y_train)
        return model
    
    elif model_type == "RandomForest":
        # Reshape data for Random Forest
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_reshaped, y_train)
        return model
    
    elif model_type == "KNN":
        # Reshape data for KNN
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_reshaped, y_train)
        return model
    
    # Hybrid Models
    elif model_type == "CNN_LSTM":
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_cnn_lstm_model(input_shape)
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        return model
    
    elif model_type == "ARIMA_XGBoost":
        # Simplified hybrid implementation
        try:
            arima_model = ARIMA(y_train, order=(5,1,0))
            arima_fit = arima_model.fit()
            arima_pred = arima_fit.predict(start=0, end=len(y_train)-1)
        except Exception as e:
            logger.error(f"ARIMA model failed: {str(e)}")
            logger.info("Falling back to simpler ARIMA model")
            arima_model = ARIMA(y_train, order=(1,0,0))
            arima_fit = arima_model.fit()
            arima_pred = arima_fit.predict(start=0, end=len(y_train)-1)
        
        # Use ARIMA predictions as features for XGBoost
        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        # Ensure arima_pred has the same length as X_reshaped
        arima_pred = arima_pred[-X_reshaped.shape[0]:]
        X_with_arima = np.column_stack((X_reshaped, arima_pred))
        
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_with_arima, y_train)
        
        return {"arima": arima_fit, "xgboost": xgb_model}
    
    elif model_type == "LSTM_Attention":
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_attention_model(input_shape)
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)
        return model
    
    elif model_type == "LSTM_XGBoost":
        return build_lstm_xgboost_model(X_train, y_train)
    
    else:
        logger.error(f"Unknown model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def create_metrics_comparison_plot(model_metrics):
    """Create comparison plot for model metrics"""
    logger.info("Creating metrics comparison plot")
    
    metrics = ['rmse', 'mae', 'r2']
    models = list(model_metrics.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        values = [model_metrics[model][metric] for model in models]
        
        # For R², higher is better, so we'll use a different color scheme
        if metric == 'r2':
            colors = ['green' if v > 0 else 'red' for v in values]
            axes[i].bar(models, values, color=colors)
            axes[i].set_title(f'R² Score (higher is better)')
        else:
            axes[i].bar(models, values, color='blue')
            axes[i].set_title(f'{metric.upper()} (lower is better)')
        
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(metric.upper())
        axes[i].set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    logger.info("Metrics comparison plot created successfully")
    return plot_data

def create_confusion_matrix(y_true, y_pred, model_name):
    """Create confusion matrix for trend prediction (up/down)"""
    logger.info(f"Creating confusion matrix for {model_name}")
    
    # Handle potential length mismatch
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Convert to trends (up/down)
    y_true_trend = np.where(np.diff(np.append([0], y_true)) > 0, 1, 0)
    y_pred_trend = np.where(np.diff(np.append([0], y_pred)) > 0, 1, 0)
    
    # Create confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true_trend)):
        cm[y_true_trend[i]][y_pred_trend[i]] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Trend Confusion Matrix - {model_name}')
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    # Calculate accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() if cm.sum() > 0 else 0
    logger.info(f"Trend prediction accuracy for {model_name}: {accuracy:.4f}")
    
    return plot_data

def predict_future(model, model_type, last_sequence, predict_year, symbol_data):
    """Predict future stock prices for the entire year"""
    # Get all dates for the prediction year
    start_date = f"{predict_year}-01-01"
    end_date = f"{predict_year}-12-31"
    
    # Create a date range for predictions
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Initialize predictions array
    predictions = []
    
    # Current sequence to predict from
    current_sequence = last_sequence.copy()
    
    try:
        # Make predictions for each date in the range
        for _ in range(len(prediction_dates)):
            if model_type in ["LSTM", "HATR", "CNN_LSTM", "LSTM_Attention"]:
                pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)
            elif model_type in ["ARIMA_XGBoost", "LSTM_XGBoost"]:
                # Handle hybrid models
                if model_type == "ARIMA_XGBoost":
                    arima_pred = model["arima"].forecast(steps=1)
                    X_reshaped = current_sequence.reshape(1, -1)
                    X_with_arima = np.column_stack((X_reshaped, [arima_pred[0]]))
                    pred = model["xgboost"].predict(X_with_arima)
                else:  # LSTM_XGBoost
                    lstm_pred = model["lstm"].predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]), verbose=0)
                    X_reshaped = current_sequence.reshape(1, -1)
                    X_with_lstm = np.column_stack((X_reshaped, lstm_pred))
                    pred = model["xgboost"].predict(X_with_lstm)
            elif model_type == "ARIMA":
                pred = model.forecast(steps=1)
            else:  # XGBoost, LightGBM, SVM, RandomForest, KNN
                X_reshaped = current_sequence.reshape(1, -1)
                pred = model.predict(X_reshaped)
            
            # Add prediction to results
            predictions.append(pred[0])
            
            # Update sequence for next prediction (roll the window)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            
            # Convert the prediction back to the original range
            if len(current_sequence.shape) == 2:  # 2D array
                last_row = current_sequence[-1].copy()
                last_row[-1] = pred[0]  # Replace the last value (target) with prediction
                current_sequence[-1] = last_row
            else:  # For ARIMA or other 1D models
                current_sequence[-1] = pred[0]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        # If an error occurs, return data up to the point of failure
        if not predictions:
            # If no predictions were made, create a dummy prediction
            predictions = [last_sequence[-1, -1]] * len(prediction_dates)
            logger.warning("Using dummy predictions due to error")
    
    # Ensure we have predictions for all dates
    if len(predictions) < len(prediction_dates):
        # Pad with last prediction
        last_pred = predictions[-1] if predictions else last_sequence[-1, -1]
        predictions.extend([last_pred] * (len(prediction_dates) - len(predictions)))
    
    # Scale predictions back to original range
    try:
        original_predictions = scaler.inverse_transform(np.array([[0] * (current_sequence.shape[1] - 1) + [p] for p in predictions]))[:, -1]
    except Exception as e:
        logger.error(f"Error scaling predictions: {str(e)}")
        # Fallback to unscaled predictions
        original_predictions = np.array(predictions)
    
    return prediction_dates, original_predictions

def calculate_trends(prices):
    """Calculate price trends"""
    trends = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            trends.append(1)  # Upward trend
        elif prices[i] < prices[i-1]:
            trends.append(-1)  # Downward trend
        else:
            trends.append(0)  # No change
    return trends

def create_price_plot(dates, prices, symbol):
    """Create price plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, 'b-')
    plt.title(f'Stock Price Prediction for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

def create_trend_plot(dates, trends, symbol):
    """Create trend plot"""
    plt.figure(figsize=(10, 4))
    plt.bar(dates[1:], trends, color=['red' if t < 0 else 'green' for t in trends])
    plt.title(f'Stock Price Trends for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Trend (Up/Down)')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_data

@app.route('/')
def index():
    try:
        # Discover available datasets
        dataset_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        # Define model types
        individual_models = ["LSTM", "HATR", "XGBoost"]
        if LIGHTGBM_AVAILABLE:
            individual_models.append("LightGBM")
        individual_models.extend(["ARIMA", "SVM", "RandomForest", "KNN"])
        
        hybrid_models = ["CNN_LSTM", "ARIMA_XGBoost", "LSTM_Attention", "LSTM_XGBoost"]
        
        return render_template('index.html', 
                            datasets=dataset_files,
                            individual_models=individual_models,
                            hybrid_models=hybrid_models)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        model_category = data['model_category']
        model_type = data['model_type']
        dataset_path = data['dataset']
        predict_year = int(data['predict_year'])
        
        # Check if requested model is LightGBM but not available
        if model_type == "LightGBM" and not LIGHTGBM_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'LightGBM is not available, please install it or select another model'
            }), 400
        
        # Load and preprocess data
        df = load_dataset(dataset_path)
        
        # Get unique symbols if available
        symbols = df['Symbol'].unique() if 'Symbol' in df.columns else ["STOCK"]
        symbol = symbols[0]  # Use first symbol for simplicity
        
        # Define feature columns
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume'] if 'Volume' in df.columns else ['Open', 'High', 'Low', 'Close']
        
        # Preprocess data for the selected symbol
        X, y = preprocess_data(df, symbol, feature_columns)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Data split - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
        
        # Train the selected model
        model = train_model(model_type, X_train, y_train, X_test, y_test)
        
        # Make predictions for the test set to create confusion matrix
        if model_type in ["LSTM", "HATR", "CNN_LSTM", "LSTM_Attention"]:
            y_pred_test = model.predict(X_test, verbose=0)
        elif model_type in ["ARIMA_XGBoost", "LSTM_XGBoost"]:
            # Handle hybrid models
            if model_type == "ARIMA_XGBoost":
                arima_pred = model["arima"].forecast(steps=len(y_test))
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                # Make sure arima_pred has the right length
                arima_pred_adjusted = arima_pred[:len(X_test_reshaped)]
                if len(arima_pred_adjusted) < len(X_test_reshaped):
                    # Pad with the last prediction
                    arima_pred_adjusted = np.append(arima_pred_adjusted, 
                                                 [arima_pred_adjusted[-1]] * (len(X_test_reshaped) - len(arima_pred_adjusted)))
                X_with_arima = np.column_stack((X_test_reshaped, arima_pred_adjusted))
                y_pred_test = model["xgboost"].predict(X_with_arima)
            else:  # LSTM_XGBoost
                lstm_pred = model["lstm"].predict(X_test, verbose=0)
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                X_with_lstm = np.column_stack((X_test_reshaped, lstm_pred))
                y_pred_test = model["xgboost"].predict(X_with_lstm)
        elif model_type == "ARIMA":
            y_pred_test = model.forecast(steps=len(y_test))
        else:  # XGBoost, LightGBM, SVM, RandomForest, KNN
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            y_pred_test = model.predict(X_test_reshaped)
        
        # Create confusion matrix
        confusion_matrix_plot = create_confusion_matrix(y_test, y_pred_test, model_type)
        
        # Create metrics comparison plot if we have metrics for multiple models
        metrics_comparison_plot = create_metrics_comparison_plot(model_metrics) if len(model_metrics) > 1 else None
        
        # Make predictions for the future year
        logger.info(f"Making predictions for year {predict_year}")
        last_sequence = X[-1]



# Get symbol data for inverse transformation
        if 'Symbol' in df.columns:
            symbol_data = df[df['Symbol'] == symbol]
        else:
            symbol_data = df
        
        # Predict future prices
        prediction_dates, predictions = predict_future(model, model_type, last_sequence, predict_year, symbol_data)
        logger.info(f"Generated {len(predictions)} predictions for {predict_year}")
        
        # Calculate trends
        trends = calculate_trends(predictions)
        logger.info(f"Calculated {len(trends)} trend indicators")
        
        # Create plots
        price_plot = create_price_plot(prediction_dates, predictions, symbol)
        trend_plot = create_trend_plot(prediction_dates, trends, symbol)
        
        response_data = {
            'symbol': symbol,
            'price_plot': price_plot,
            'trend_plot': trend_plot,
            'confusion_matrix': confusion_matrix_plot,
            'status': 'success'
        }
        
        if metrics_comparison_plot:
            response_data['metrics_comparison'] = metrics_comparison_plot
        
        logger.info("Training and prediction completed successfully")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Create a simple HTML template for the UI
@app.route('/create_template', methods=['GET'])
def create_template():
    """Create the HTML template file if it doesn't exist"""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    template_path = os.path.join(template_dir, 'index.html')
    
    # Only create if it doesn't exist
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Price Prediction</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body { padding: 30px; }
        .container { max-width: 1200px; }
        .chart-container { margin-top: 20px; }
        .loading { display: none; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Stock Price Prediction System</h1>
        
        <div class="card">
            <div class="card-header">
                <h4>Training Configuration</h4>
            </div>
            <div class="card-body">
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="dataset">Select Dataset:</label>
                        <select class="form-control" id="dataset" name="dataset">
                            {% for dataset in datasets %}
                            <option value="{{ dataset }}">{{ dataset }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="modelCategory">Model Category:</label>
                        <select class="form-control" id="modelCategory" name="modelCategory">
                            <option value="individual">Individual Models</option>
                            <option value="hybrid">Hybrid Models</option>
                        </select>
                    </div>
                    
                    <div class="form-group" id="individualModelsGroup">
                        <label for="individualModel">Individual Model:</label>
                        <select class="form-control" id="individualModel" name="individualModel">
                            {% for model in individual_models %}
                            <option value="{{ model }}">{{ model }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group" id="hybridModelsGroup" style="display: none;">
                        <label for="hybridModel">Hybrid Model:</label>
                        <select class="form-control" id="hybridModel" name="hybridModel">
                            {% for model in hybrid_models %}
                            <option value="{{ model }}">{{ model }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="predictYear">Prediction Year:</label>
                        <input type="number" class="form-control" id="predictYear" name="predictYear" value="2024" min="2023" max="2030">
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Train Model and Predict</button>
                </form>
            </div>
        </div>
        
        <div class="loading text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="sr-only">Loading...</span>
            </div>
            <p>Training model and generating predictions. This may take a few minutes...</p>
        </div>
        
        <div id="resultsContainer" style="display: none;">
            <h2 class="mt-4">Prediction Results</h2>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5>Price Predictions</h5>
                </div>
                <div class="card-body">
                    <img id="pricePlot" class="img-fluid" alt="Price Predictions">
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5>Trend Analysis</h5>
                </div>
                <div class="card-body">
                    <img id="trendPlot" class="img-fluid" alt="Trend Analysis">
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">
                    <h5>Model Evaluation</h5>
                </div>
                <div class="card-body">
                    <h6>Trend Prediction Accuracy</h6>
                    <img id="confusionMatrix" class="img-fluid" alt="Confusion Matrix">
                    
                    <div id="metricsComparison" style="display: none;">
                        <h6 class="mt-4">Metrics Comparison</h6>
                        <img id="metricsPlot" class="img-fluid" alt="Metrics Comparison">
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    
    <script>
        $(document).ready(function() {
            // Toggle model selection based on category
            $('#modelCategory').change(function() {
                if ($(this).val() === 'individual') {
                    $('#individualModelsGroup').show();
                    $('#hybridModelsGroup').hide();
                } else {
                    $('#individualModelsGroup').hide();
                    $('#hybridModelsGroup').show();
                }
            });
            
            // Form submission
            $('#predictionForm').submit(function(e) {
                e.preventDefault();
                
                // Show loading indicator
                $('.loading').show();
                $('#resultsContainer').hide();
                
                // Get form data
                const dataset = $('#dataset').val();
                const modelCategory = $('#modelCategory').val();
                const modelType = modelCategory === 'individual' ? $('#individualModel').val() : $('#hybridModel').val();
                const predictYear = $('#predictYear').val();
                
                // Send AJAX request
                $.ajax({
                    url: '/train',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({
                        model_category: modelCategory,
                        model_type: modelType,
                        dataset: dataset,
                        predict_year: predictYear
                    }),
                    success: function(response) {
                        // Hide loading indicator
                        $('.loading').hide();
                        
                        // Display results
                        $('#pricePlot').attr('src', 'data:image/png;base64,' + response.price_plot);
                        $('#trendPlot').attr('src', 'data:image/png;base64,' + response.trend_plot);
                        $('#confusionMatrix').attr('src', 'data:image/png;base64,' + response.confusion_matrix);
                        
                        if (response.metrics_comparison) {
                            $('#metricsPlot').attr('src', 'data:image/png;base64,' + response.metrics_comparison);
                            $('#metricsComparison').show();
                        } else {
                            $('#metricsComparison').hide();
                        }
                        
                        $('#resultsContainer').show();
                    },
                    error: function(xhr, status, error) {
                        // Hide loading indicator
                        $('.loading').hide();
                        
                        // Show error message
                        alert('Error: ' + xhr.responseJSON.message || 'An unknown error occurred');
                    }
                });
            });
        });
    </script>
</body>
</html>
            """)
        logger.info(f"Created template file at {template_path}")
    
    return "Template created successfully!"

# Add a route to download sample data if needed
@app.route('/download_sample', methods=['GET'])
def download_sample():
    """Create a sample stock data file for testing"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create a sample dataset
    start_date = datetime(2015, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    # Generate dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Only weekdays
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Generate stock data
    np.random.seed(42)
    
    # Initial price
    initial_price = 100.0
    
    # Create empty lists
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    # Generate data with some realistic patterns
    price = initial_price
    for i in range(len(dates)):
        # Daily change percentage
        daily_change = np.random.normal(0.0005, 0.015)
        
        # Open price
        open_price = price
        opens.append(open_price)
        
        # Close price
        close_price = open_price * (1 + daily_change)
        closes.append(close_price)
        
        # High and low prices
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        highs.append(high_price)
        lows.append(low_price)
        
        # Volume
        volume = int(np.random.normal(1000000, 300000))
        volumes.append(max(1000, volume))
        
        # Set price for next day
        price = close_price
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    # Save to CSV
    sample_file = 'sample_stock_data.csv'
    df.to_csv(sample_file, index=False)
    
    return f"Sample stock data created at {sample_file}"

if __name__ == '__main__':
    # Create template directory if it doesn't exist
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    # Check if template exists, if not create it
    template_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(template_path):
        logger.info("Template file not found, creating it...")
        with open(template_path, 'w') as f:
            # Write simplified template content
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Price Prediction</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container py-4">
        <h1>Stock Price Prediction System</h1>
        <p>Please visit <a href="/create_template">/create_template</a> to create the full template.</p>
    </div>
</body>
</html>
            """)
    
    # Create sample data if no CSV files exist
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        logger.info("No CSV files found, creating sample data...")
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create a simple sample dataset
        dates = pd.date_range(start='2015-01-01', end='2022-12-31', freq='B')
        np.random.seed(42)
        
        # Initial price
        price = 100.0
        
        # Create data with simple random walk
        data = []
        for i in range(len(dates)):
            day_change = np.random.normal(0.0005, 0.015)
            open_price = price
            close_price = open_price * (1 + day_change)
            high_price = max(open_price, close_price) * 1.005
            low_price = min(open_price, close_price) * 0.995
            volume = int(np.random.normal(1000000, 300000))
            
            data.append([dates[i], open_price, high_price, low_price, close_price, volume])
            price = close_price
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Save to CSV
        sample_file = 'sample_stock_data.csv'
        df.to_csv(sample_file, index=False)
        logger.info(f"Created sample stock data at {sample_file}")
    
    logger.info("Starting Flask application")
    app.run(debug=True)