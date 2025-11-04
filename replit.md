# Air Quality Predictor

## Overview

This is a Streamlit-based web application for air quality prediction and analysis. The application uses **three advanced machine learning models** (Random Forest, XGBoost, and LSTM) to forecast six air quality metrics (PM2.5, PM10, CO2, CO, Temperature, Humidity). It provides comprehensive model comparison visualizations using interactive Plotly charts and integrates with Supabase as a backend database for storing and retrieving air quality data.

## Recent Changes (November 2025)

- Added **XGBoost** and **LSTM** models for enhanced prediction accuracy
- Implemented multi-model comparison framework
- Added comprehensive model performance comparison visualizations
- Enhanced prediction interface to show results from all three models side-by-side
- Improved time-series forecasting with iterative multi-step prediction

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - chosen for rapid development of data-driven web applications with minimal frontend code
- **Visualization**: Plotly (graph_objects and express) - provides interactive, publication-quality charts for time-series data and predictions
- **Data Processing**: Pandas and NumPy - standard Python libraries for efficient data manipulation and numerical operations

### Backend Architecture
- **Application Server**: Streamlit's built-in server handles HTTP requests and manages user sessions
- **Machine Learning Models**: 
  - Random Forest Regressor - ensemble method for robust predictions with non-linear patterns
  - XGBoost - gradient boosting algorithm optimized for speed and performance
  - LSTM (Long Short-Term Memory) - deep learning recurrent neural network for time-series forecasting
  - All models trained on time-based features and lagged values with iterative multi-step forecasting
- **Model Training**: Train/test split approach (80/20) with temporal order preservation (shuffle=False)
- **Feature Engineering**: Time-based features (hour, day, month) + lag features + rolling statistics
- **Evaluation Metrics**: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)

### Data Storage
- **Database**: Supabase (PostgreSQL-based)
  - Chosen for managed PostgreSQL with real-time capabilities and built-in authentication
  - Python client library (`supabase-py`) provides simple async/await interface
  - Stores historical air quality measurements for model training and analysis

### Configuration Management
- **Secrets Management**: Dual-layer approach for flexibility
  1. Primary: Streamlit secrets (`.streamlit/secrets.toml`) - recommended for deployment
  2. Fallback: Environment variables - useful for local development and CI/CD pipelines
- **Rationale**: Provides flexibility across different deployment environments while maintaining security

### Error Handling
- Graceful degradation when credentials are missing with user-friendly setup instructions
- Warning suppression for cleaner user experience (scikit-learn deprecation warnings)

## External Dependencies

### Third-Party Services
- **Supabase**: Cloud-hosted PostgreSQL database with real-time features
  - Purpose: Persistent storage for air quality time-series data
  - Authentication: Anonymous key-based access
  - Connection: REST API via Python client library

### Python Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive data visualization
- **supabase**: Supabase client library for database connectivity
- **scikit-learn**: Machine learning models and evaluation metrics
  - RandomForestRegressor: Ensemble prediction model
  - MinMaxScaler: Data normalization for LSTM
  - Model evaluation utilities (MAE, MSE, train_test_split)
- **xgboost**: XGBoost gradient boosting library for high-performance predictions
- **tensorflow/keras**: Deep learning framework for LSTM model implementation
  - Sequential model architecture
  - LSTM and Dense layers for time-series forecasting
  - EarlyStopping callback for training optimization

### Environment Requirements
- Python 3.x runtime
- Access to Supabase instance with valid URL and anonymous key
- Internet connectivity for Supabase API calls and Plotly rendering