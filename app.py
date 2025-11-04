import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Supabase Configuration
# Try Streamlit secrets first, then environment variables
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
    SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY"))
except:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Validate credentials are set
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("⚠️ Supabase credentials not configured!")
    st.info("""
    Please configure Supabase credentials using one of these methods:
    
    **Option 1: Streamlit Secrets (Recommended)**
    - Create `.streamlit/secrets.toml` with:
    ```toml
    SUPABASE_URL = "your-url"
    SUPABASE_ANON_KEY = "your-key"
    ```
    
    **Option 2: Environment Variables**
    - Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables
    """)
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🌍 Air Quality Prediction Dashboard")
st.markdown("Predict future air quality metrics (PM2.5, PM10, CO2, CO, Temperature, Humidity) using machine learning models trained on historical data")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None

# Fetch data from Supabase
@st.cache_data(ttl=300)
def fetch_data():
    """Fetch air quality data from Supabase"""
    try:
        supabase = init_supabase()
        if supabase is None:
            return None
        
        response = supabase.table('airquality').select('*').order('created_at', desc=False).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Sort by timestamp
            df = df.sort_values('created_at').reset_index(drop=True)
            
            return df
        else:
            st.warning("No data found in the database")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Feature engineering for time series
def create_time_features(df):
    """Create time-based features for better predictions"""
    df = df.copy()
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.dayofweek
    df['day_of_month'] = df['created_at'].dt.day
    df['month'] = df['created_at'].dt.month
    
    # Create lag features for each metric
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    for metric in metrics:
        df[f'{metric}_lag1'] = df[metric].shift(1)
        df[f'{metric}_lag2'] = df[metric].shift(2)
        df[f'{metric}_lag3'] = df[metric].shift(3)
        df[f'{metric}_rolling_mean_3'] = df[metric].rolling(window=3, min_periods=1).mean()
        df[f'{metric}_rolling_std_3'] = df[metric].rolling(window=3, min_periods=1).std()
    
    # Drop rows with NaN values from lag features
    df = df.dropna()
    
    return df

# Train models for each metric
@st.cache_resource
def train_models(df):
    """Train Random Forest models for each air quality metric"""
    if df is None or len(df) < 10:
        return None
    
    # Create features
    df_features = create_time_features(df)
    
    if len(df_features) < 5:
        st.error("Not enough data for training after feature engineering")
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    models = {}
    performance = {}
    
    # Base features (time features + all lag features)
    base_features = ['hour', 'day_of_week', 'day_of_month', 'month']
    
    for metric in metrics:
        # Features for this metric
        lag_features = [col for col in df_features.columns if metric in col and col != metric]
        features = base_features + lag_features
        
        X = df_features[features]
        y = df_features[metric]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        models[metric] = {
            'model': model,
            'features': features,
            'scaler': None
        }
        
        performance[metric] = {
            'MAE': mae,
            'RMSE': rmse,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    return models, performance, df_features

# Make predictions
def predict_future(models, df_features, hours_ahead=1):
    """Predict future values for all metrics using iterative multi-step forecasting"""
    if models is None or df_features is None:
        return None
    
    metrics = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
    
    # Get the latest data point
    current_data = df_features.iloc[-1].copy()
    last_timestamp = current_data['created_at']
    
    # Initialize predictions dictionary
    step_predictions = {}
    
    # Iteratively predict for each hour
    for step in range(1, hours_ahead + 1):
        # Calculate timestamp for this step
        step_timestamp = last_timestamp + timedelta(hours=step)
        
        # Update time features for this step
        time_features = {
            'hour': step_timestamp.hour,
            'day_of_week': step_timestamp.dayofweek,
            'day_of_month': step_timestamp.day,
            'month': step_timestamp.month
        }
        
        # Predict each metric for this step
        step_predictions = {}
        for metric in metrics:
            model_data = models[metric]
            model = model_data['model']
            features = model_data['features']
            
            # Prepare features for prediction
            X_pred = []
            for feature in features:
                if feature in time_features:
                    X_pred.append(time_features[feature])
                else:
                    X_pred.append(current_data[feature])
            
            # Make prediction for this metric
            pred_value = model.predict([X_pred])[0]
            step_predictions[metric] = pred_value
        
        # Roll forward: Update current_data with predictions for next iteration
        # Update lag features based on predicted values
        for metric in metrics:
            # Capture old lag values before overwriting
            old_lag1 = current_data[f'{metric}_lag1']
            old_lag2 = current_data[f'{metric}_lag2']
            
            # Shift lag values
            current_data[f'{metric}_lag3'] = current_data[f'{metric}_lag2']
            current_data[f'{metric}_lag2'] = old_lag1
            current_data[f'{metric}_lag1'] = step_predictions[metric]
            
            # Update rolling statistics using distinct recent values
            recent_values = [
                step_predictions[metric],
                old_lag1,
                old_lag2
            ]
            current_data[f'{metric}_rolling_mean_3'] = np.mean(recent_values)
            current_data[f'{metric}_rolling_std_3'] = np.std(recent_values)
        
        # Update timestamp
        current_data['created_at'] = step_timestamp
    
    # Return the final predictions (for the target hour)
    predictions = {metric: step_predictions[metric] for metric in metrics}
    predictions['timestamp'] = last_timestamp + timedelta(hours=hours_ahead)
    
    return predictions

# Main app
def main():
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Fetch data
    with st.spinner("Fetching data from Supabase..."):
        df = fetch_data()
    
    if df is None or len(df) == 0:
        st.error("Unable to load data from database. Please check your connection and try again.")
        return
    
    st.sidebar.success(f"✅ Loaded {len(df)} records")
    st.sidebar.markdown(f"**Date Range:** {df['created_at'].min().strftime('%Y-%m-%d')} to {df['created_at'].max().strftime('%Y-%m-%d')}")
    
    # Prediction settings
    st.sidebar.subheader("Prediction Settings")
    forecast_hours = st.sidebar.slider(
        "Forecast Hours Ahead",
        min_value=1,
        max_value=24,
        value=6,
        help="Select how many hours into the future to predict"
    )
    
    # Train models
    with st.spinner("Training machine learning models..."):
        result = train_models(df)
        
        if result is None:
            st.error("Unable to train models. Not enough data available.")
            return
        
        models, performance, df_features = result
    
    st.sidebar.success("✅ Models trained successfully")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔮 Predictions", "📈 Historical Trends", "📉 Model Performance"])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Recent Data")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        latest = df.iloc[-1]
        
        with col1:
            st.metric("Temperature", f"{latest['temperature']:.1f}°C")
        with col2:
            st.metric("Humidity", f"{latest['humidity']:.1f}%")
        with col3:
            st.metric("CO2", f"{latest['co2']:.0f} ppm")
        with col4:
            st.metric("CO", f"{latest['co']:.2f} ppm")
        with col5:
            st.metric("PM2.5", f"{latest['pm25']:.1f} µg/m³")
        with col6:
            st.metric("PM10", f"{latest['pm10']:.1f} µg/m³")
        
        st.subheader("Latest 50 Records")
        display_df = df[['created_at', 'temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']].tail(50)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Tab 2: Predictions
    with tab2:
        st.header("🔮 Future Predictions")
        
        predictions = predict_future(models, df_features, hours_ahead=forecast_hours)
        
        if predictions:
            st.markdown(f"### Predicted values for **{predictions['timestamp'].strftime('%Y-%m-%d %H:%M')}** ({forecast_hours} hours ahead)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🌡️ Temperature",
                    f"{predictions['temperature']:.1f}°C",
                    delta=f"{predictions['temperature'] - latest['temperature']:.1f}°C"
                )
                st.metric(
                    "💧 Humidity",
                    f"{predictions['humidity']:.1f}%",
                    delta=f"{predictions['humidity'] - latest['humidity']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "🏭 CO2",
                    f"{predictions['co2']:.0f} ppm",
                    delta=f"{predictions['co2'] - latest['co2']:.0f} ppm"
                )
                st.metric(
                    "☠️ CO",
                    f"{predictions['co']:.2f} ppm",
                    delta=f"{predictions['co'] - latest['co']:.2f} ppm"
                )
            
            with col3:
                st.metric(
                    "🌫️ PM2.5",
                    f"{predictions['pm25']:.1f} µg/m³",
                    delta=f"{predictions['pm25'] - latest['pm25']:.1f} µg/m³"
                )
                st.metric(
                    "💨 PM10",
                    f"{predictions['pm10']:.1f} µg/m³",
                    delta=f"{predictions['pm10'] - latest['pm10']:.1f} µg/m³"
                )
            
            # Prediction comparison chart
            st.subheader("Current vs Predicted Values")
            
            metrics_names = ['Temperature', 'Humidity', 'CO2', 'CO', 'PM2.5', 'PM10']
            metrics_keys = ['temperature', 'humidity', 'co2', 'co', 'pm25', 'pm10']
            
            current_values = [latest[key] for key in metrics_keys]
            predicted_values = [predictions[key] for key in metrics_keys]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=metrics_names,
                y=current_values,
                name='Current',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=metrics_names,
                y=predicted_values,
                name=f'Predicted ({forecast_hours}h)',
                marker_color='salmon'
            ))
            
            fig.update_layout(
                barmode='group',
                title='Current vs Predicted Values',
                xaxis_title='Metrics',
                yaxis_title='Value',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Historical Trends
    with tab3:
        st.header("📈 Historical Data Visualization")
        
        metric_select = st.selectbox(
            "Select Metric to Visualize",
            options=[
                ('Temperature', 'temperature'),
                ('Humidity', 'humidity'),
                ('CO2', 'co2'),
                ('CO', 'co'),
                ('PM2.5', 'pm25'),
                ('PM10', 'pm10')
            ],
            format_func=lambda x: x[0]
        )
        
        metric_name, metric_key = metric_select
        
        # Time series plot
        fig = px.line(
            df,
            x='created_at',
            y=metric_key,
            title=f'{metric_name} Over Time',
            labels={'created_at': 'Time', metric_key: metric_name}
        )
        
        fig.update_traces(line_color='#1f77b4', line_width=2)
        fig.update_layout(height=400, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[metric_key].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[metric_key].median():.2f}")
        with col3:
            st.metric("Min", f"{df[metric_key].min():.2f}")
        with col4:
            st.metric("Max", f"{df[metric_key].max():.2f}")
        
        # All metrics comparison
        st.subheader("All Metrics Over Time")
        
        fig = go.Figure()
        
        metrics = [
            ('Temperature', 'temperature', '°C'),
            ('Humidity', 'humidity', '%'),
            ('CO2', 'co2', 'ppm'),
            ('CO', 'co', 'ppm'),
            ('PM2.5', 'pm25', 'µg/m³'),
            ('PM10', 'pm10', 'µg/m³')
        ]
        
        for name, key, unit in metrics:
            # Normalize values for comparison
            normalized = (df[key] - df[key].min()) / (df[key].max() - df[key].min())
            fig.add_trace(go.Scatter(
                x=df['created_at'],
                y=normalized,
                name=name,
                mode='lines',
                hovertemplate=f'{name}: %{{customdata:.2f}} {unit}<extra></extra>',
                customdata=df[key]
            ))
        
        fig.update_layout(
            title='Normalized Comparison of All Metrics',
            xaxis_title='Time',
            yaxis_title='Normalized Value (0-1)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Model Performance
    with tab4:
        st.header("📉 Model Performance Metrics")
        
        st.markdown("""
        The models are evaluated using:
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values (lower is better)
        - **RMSE (Root Mean Squared Error)**: Square root of average squared differences (lower is better, penalizes large errors more)
        """)
        
        # Create performance dataframe
        perf_data = []
        for metric, perf in performance.items():
            perf_data.append({
                'Metric': metric.upper().replace('_', ' '),
                'MAE': f"{perf['MAE']:.3f}",
                'RMSE': f"{perf['RMSE']:.3f}",
                'Training Samples': perf['train_size'],
                'Test Samples': perf['test_size']
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        # Visualize performance
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[p['Metric'] for p in perf_data],
                y=[performance[m]['MAE'] for m in performance.keys()],
                marker_color='lightcoral',
                name='MAE'
            ))
            fig.update_layout(
                title='Mean Absolute Error by Metric',
                xaxis_title='Metric',
                yaxis_title='MAE',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[p['Metric'] for p in perf_data],
                y=[performance[m]['RMSE'] for m in performance.keys()],
                marker_color='lightblue',
                name='RMSE'
            ))
            fig.update_layout(
                title='Root Mean Squared Error by Metric',
                xaxis_title='Metric',
                yaxis_title='RMSE',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Model Details**: Random Forest Regressors with 100 trees, trained on time-based features and lagged values")

if __name__ == "__main__":
    main()
