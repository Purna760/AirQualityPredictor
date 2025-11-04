# Air Quality Predictor

## Overview

This is a Streamlit-based web application for air quality prediction and analysis. The application uses machine learning models (Random Forest and Linear Regression) to forecast air quality metrics and visualizes the results using interactive Plotly charts. It integrates with Supabase as a backend database for storing and retrieving air quality data.

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
  - Linear Regression - baseline model for comparison and simpler linear trend analysis
  - scikit-learn framework provides consistent API and evaluation metrics (MAE, MSE)
- **Model Training**: Train/test split approach for model validation

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
  - RandomForestRegressor: Primary prediction model
  - LinearRegression: Baseline comparison model
  - Model evaluation utilities (MAE, MSE, train_test_split)

### Environment Requirements
- Python 3.x runtime
- Access to Supabase instance with valid URL and anonymous key
- Internet connectivity for Supabase API calls and Plotly rendering