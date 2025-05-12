# Multi-Stock Price & Trends Prediction System

A web-based application for predicting stock prices and trend analysis using various machine learning and deep learning models. This system allows users to select from individual or hybrid models to predict stock performance based on historical data.

## Features

- **Multiple Model Support**:
  - **Individual Models**: LSTM, HATR, XGBoost, LightGBM, ARIMA, SVM, Random Forest, KNN
  - **Hybrid Models**: CNN+LSTM, ARIMA+XGBoost, LSTM+Attention, LSTM+XGBoost

- **Interactive Web Interface**:
  - Dynamic model selection based on category
  - Dataset selection
  - Prediction year configuration
  - Modern, responsive UI with animations

- **Comprehensive Visualizations**:
  - Price prediction graphs with historical context
  - Trend analysis showing upward and downward movements
  - Interactive elements and modern styling

## Installation

1. Clone this repository or download the source files

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your stock dataset CSV files in the project root directory

4. Create a `templates` folder and place `index.html` inside it

5. Create a `static` folder with a `css` subfolder and place `styles.css` inside it

## Dataset Format

The system accepts CSV files with the following attributes:
- Date
- Symbol
- Series
- Prev Close
- Open
- High
- Low
- Last
- Close
- VWAP
- Volume
- Turnover
- Trades
- Deliverable Volume
- %Deliverble

Example dataset row:
```
2000-01-03,INFOSYSTCH,EQ,14467.75,15625.0,15625.2,15625.0,15625.2,15625.2,15625.18,5137,8026657140000.001,,,
```

## Usage

1. Start the Flask server:
   ```
   python model_training_flask_server.py
   ```

2. Access the web application at:
   ```
   http://127.0.0.1:5000/
   ```

3. Select model category (Individual or Hybrid)

4. Choose a specific model type

5. Select your dataset

6. Enter the year you want to predict

7. Click "Train Model & Predict"

8. View results in the prediction graphs

## Model Information

### Individual Models

- **LSTM (Long Short-Term Memory)**: Specialized deep learning model for sequential data
- **HATR (Hierarchical Attention Temporal Recurrent)**: Advanced RNN with attention mechanisms
- **XGBoost (Extreme Gradient Boosting)**: Tree-based ensemble method
- **LightGBM (Light Gradient Boosting Machine)**: Gradient boosting framework using tree-based learning
- **ARIMA (AutoRegressive Integrated Moving Average)**: Statistical method for time series
- **SVM (Support Vector Machine)**: Classification algorithm for regression
- **Random Forest Regressor**: Ensemble method using multiple decision trees
- **KNN (K-Nearest Neighbors)**: Non-parametric pattern recognition method

### Hybrid Models

- **CNN + LSTM**: Combines convolutional and recurrent networks
- **ARIMA + XGBoost**: Statistical time series with gradient boosting
- **LSTM + Attention**: Recurrent network with attention mechanism
- **LSTM + XGBoost**: Deep learning with gradient boosting

## Performance

The system is optimized to complete model training within 10-100 seconds, depending on dataset size and model complexity.

## License

This project is available for open use and modification.

## Acknowledgements

- Built with Flask, TensorFlow, scikit-learn, and other open-source libraries
- Visualization powered by Matplotlib