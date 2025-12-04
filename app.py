import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from neuralprophet import NeuralProphet

# Page config for better appearance
st.set_page_config(
    page_title="Solar GHI Forecasting Dashboard",
    page_icon="🌞",
    layout="wide"
)

# Add custom CSS for background and styling
st.markdown("""
<style>
    /* Main background - Soft light gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Content containers */
    .main-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Sidebar styling - Soft blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6a89cc 0%, #4a69bd 100%);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        color: white;
    }
    
    /* Metric cards - Soft cards */
    .stMetric {
        background: white;
        border-left: 4px solid #4a69bd;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        color: #2d3748 !important;
    }
    
    .stMetric label {
        color: #4a5568 !important;
        font-weight: 600;
    }
    
    .stMetric div {
        color: #2d3748 !important;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    /* Button styling - Primary blue */
    .stButton > button {
        background: linear-gradient(135deg, #4a69bd 0%, #6a89cc 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3c56a0 0%, #5a75b8 100%);
        box-shadow: 0 4px 12px rgba(74, 105, 189, 0.3);
    }
    
    /* Header styling */
    h1 {
        color: #2d3748 !important;
        border-bottom: 2px solid #4a69bd;
        padding-bottom: 10px;
    }
    
    h2, h3, .stSubheader {
        color: #4a5568 !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: white !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Input fields */
    .stDateInput > div > div {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Success messages */
    .stAlert > div {
        background-color: rgba(72, 187, 120, 0.1);
        border: 1px solid rgba(72, 187, 120, 0.3);
        color: #276749;
    }
    
    /* Info messages */
    .stInfo > div {
        background-color: rgba(66, 153, 225, 0.1);
        border: 1px solid rgba(66, 153, 225, 0.3);
        color: #2c5282;
    }
    
    /* Warning messages */
    .stWarning > div {
        background-color: rgba(237, 137, 54, 0.1);
        border: 1px solid rgba(237, 137, 54, 0.3);
        color: #9c4221;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
        border-radius: 10px;
        padding: 5px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
    }
    
    /* Plotly chart background */
    .js-plotly-plot {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Table headers */
    .dataframe th {
        background-color: #f7fafc !important;
        color: #4a5568 !important;
        font-weight: 600 !important;
    }
    
    /* Table rows */
    .dataframe tr:nth-child(even) {
        background-color: #f7fafc !important;
    }
    
    .dataframe tr:hover {
        background-color: #edf2f7 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jomhelcallueng/solar-forecast-dashboard/refs/heads/main/20251110_Lallo_hourly.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['SZA'])
    df = df.rename(columns={
        'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour',
        'ALLSKY_SFC_SW_DWN': 'ghi', 'ALLSKY_SFC_SW_DNI': 'dni', 'ALLSKY_SFC_SW_DIFF': 'dhi',
        'T2M': 'temperature', 'T2MDEW': 'dew_point', 'RH2M': 'relative_humidity',
        'PRECTOTCORR': 'precipitation', 'PS': 'surface_pressure', 'WS10M': 'wind_speed'
    })
    # Create datetime column
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    return df

df = load_data()

# Create a container for all content
main_container = st.container()

with main_container:
    # Sidebar for global controls
    with st.sidebar:
        st.title("🌞 Navigation")
        st.markdown("---")
        section = st.radio(
            "Choose Section:",
            ["📊 Data Exploration", "🤖 GHI Forecasting"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        
        if section == "🤖 GHI Forecasting":
            st.header("Model Configuration")
            st.write("**Best Parameters:**")
            st.write(f"- Lags: 48 hours")
            st.write(f"- Learning Rate: 0.01")
            st.write(f"- Changepoints: 5")
    
    # ============================
    # 1. DATA EXPLORATION SECTION
    # ============================
    if section == "📊 Data Exploration":
        st.title("📊 Solar Data Exploration")
        st.markdown("---")
        
        # Single date selection with calendar
        st.subheader("📅 Select a Date to View")
        
        # Get date range from data
        min_date = df['datetime'].min().date()
        max_date = df['datetime'].max().date()
        
        # Use date_input for calendar-like selection
        selected_date = st.date_input(
            "Choose a date:",
            value=max_date,  # Default to most recent
            min_value=min_date,
            max_value=max_date,
            key="explore_date"
        )
        
        # Filter data for selected date
        selected_date_dt = pd.Timestamp(selected_date)
        daily_df = df[df['datetime'].dt.date == selected_date].copy()
        
        if len(daily_df) > 0:
            # Daily summary
            st.subheader(f"🌤️ Daily Summary for {selected_date}")
            
            # Weather metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_temp = daily_df['temperature'].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f} °C")
            
            with col2:
                avg_humidity = daily_df['relative_humidity'].mean()
                st.metric("Avg Humidity", f"{avg_humidity:.1f}%")
            
            with col3:
                total_rain = daily_df['precipitation'].sum()
                st.metric("Total Rainfall", f"{total_rain:.1f} mm")
            
            with col4:
                avg_wind = daily_df['wind_speed'].mean()
                st.metric("Avg Wind Speed", f"{avg_wind:.1f} m/s")
            
            # GHI statistics for the day
            st.subheader("☀️ GHI Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_ghi = daily_df['ghi'].sum()
                st.metric("Total Daily GHI", f"{total_ghi:.0f} W/m²")
            
            with col2:
                avg_ghi = daily_df['ghi'].mean()
                st.metric("Average GHI", f"{avg_ghi:.1f} W/m²")
            
            with col3:
                max_ghi = daily_df['ghi'].max()
                max_hour = daily_df.loc[daily_df['ghi'].idxmax(), 'datetime'].strftime('%H:%M')
                st.metric("Peak GHI", f"{max_ghi:.1f} W/m²")
                st.caption(f"at {max_hour}")
            
            with col4:
                min_ghi = daily_df['ghi'].min()
                min_hour = daily_df.loc[daily_df['ghi'].idxmin(), 'datetime'].strftime('%H:%M')
                st.metric("Minimum GHI", f"{min_ghi:.1f} W/m²")
                st.caption(f"at {min_hour}")
            
            # Hourly GHI plot
            st.subheader("📈 Hourly GHI Profile")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_df['datetime'].dt.strftime('%H:%M'),
                y=daily_df['ghi'],
                mode='lines+markers',
                name='GHI',
                line=dict(color='orange', width=3),
                marker=dict(size=6, color='darkorange'),
                hovertemplate='<b>Time:</b> %{x}<br><b>GHI:</b> %{y:.1f} W/m²<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Hourly GHI on {selected_date}',
                xaxis_title='Hour of Day',
                yaxis_title='GHI (W/m²)',
                hovermode='x unified',
                template='plotly_white',
                height=400,
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                paper_bgcolor='rgba(255, 255, 255, 0.9)',
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, 24, 3)),
                    ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)],
                    gridcolor='lightgray'
                ),
                yaxis=dict(gridcolor='lightgray')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
             # Data table for the day - CLEANER VERSION
            st.subheader("📋 Hourly Data Table")
            
            # Format for display
            display_df = daily_df.copy()
            display_df['Time'] = display_df['datetime'].dt.strftime('%H:%M')
            
            # Select only the measurement columns (skip year/month/day/hour since we have datetime)
            display_cols = ['Time', 'ghi', 'dni', 'dhi', 'temperature', 
                           'dew_point', 'relative_humidity', 'precipitation', 
                           'surface_pressure', 'wind_speed']
            
            # Rename for clarity
            display_df = display_df[display_cols].rename(columns={
                'ghi': 'GHI (W/m²)',
                'dni': 'DNI (W/m²)',
                'dhi': 'DHI (W/m²)',
                'temperature': 'Temp (°C)',
                'dew_point': 'Dew Point (°C)',
                'relative_humidity': 'Humidity (%)',
                'precipitation': 'Rain (mm)',
                'surface_pressure': 'Pressure (kPa)',
                'wind_speed': 'Wind (m/s)'
            })
            
            st.dataframe(
                display_df.round(2),
                use_container_width=True,
                height=400
            )
            
            # Download button for daily data
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Daily Data (CSV)",
                data=csv,
                file_name=f"solar_data_{selected_date}.csv",
                mime="text/csv",
            )
            
        else:
            st.warning(f"No data available for {selected_date}")
    
    # ============================
    # 2. FORECASTING SECTION
    # ============================
    else:
        st.title("🤖 Neural Prophet Forecasting with 90% Confidence Intervals")
        st.markdown("---")
        
        # Use fixed parameters
        n_lags = 48
        learning_rate = 0.01
        n_changepoints = 5
        confidence_level = 0.90  # Fixed at 90%
        alpha = 1 - confidence_level  # 0.10
        quantile_list = [alpha/2, 1-alpha/2]  # [0.05, 0.95]
        method = "naive"
        
        # Prepare data for Prophet
        with st.spinner("Preparing data..."):
            prophet_df = df.copy()
            prophet_df = prophet_df.set_index('datetime')
            prophet_df = prophet_df.rename(columns={'ghi': 'y'})
            prophet_df['ds'] = prophet_df.index
            
            # Train-test split (80% train, 10% calibration, 10% test)
            total_size = len(prophet_df)
            train_size = int(0.8 * total_size)
            cal_size = int(0.1 * total_size)
            
            train_df = prophet_df.iloc[:train_size]
            cal_df = prophet_df.iloc[train_size:train_size + cal_size]
            test_df = prophet_df.iloc[train_size + cal_size:]
            
            st.info(f"**Data split:** Train={len(train_df)} hours, Calibration={len(cal_df)} hours, Test={len(test_df)} hours")
        
        # Cache the trained model to avoid retraining
        if 'neural_prophet_model' not in st.session_state:
            st.session_state.neural_prophet_model = None
            st.session_state.neural_prophet_forecast = None
            st.session_state.neural_prophet_results = None
        
        # Train model button
        if st.button("🚀 Train Neural Prophet Model with 90% CI", type="primary"):
            
            with st.spinner(f"Training model with 90% confidence intervals..."):
                # Start timing
                start = time.perf_counter()
                
                # Model with quantiles for confidence intervals
                m2 = NeuralProphet(
                    growth="off",
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    n_lags=n_lags,
                    learning_rate=learning_rate,
                    n_changepoints=n_changepoints,
                    quantiles=quantile_list,  # For 90% CI
                )
                
                # Train on training data
                metrics = m2.fit(train_df[['ds', 'y']], freq="H", progress=None)
                train_time = time.perf_counter() - start
                
                # Get conformal predictions with confidence intervals
                forecast = m2.conformal_predict(
                    test_df[['ds', 'y']],
                    calibration_df=cal_df[['ds', 'y']],
                    alpha=alpha,
                    method=method,
                    plotting_backend=None,
                    show_all_PI=False,
                )
                
                # Make sure we have predictions
                if forecast is None or len(forecast) == 0:
                    st.error("Failed to generate predictions. Trying fallback...")
                    # Try regular predict
                    future = m2.make_future_dataframe(train_df[['ds', 'y']], periods=len(test_df))
                    forecast = m2.predict(future)
                
                # Extract predictions - handle different column names
                y_pred = forecast['yhat1'].values if 'yhat1' in forecast.columns else None
                
                if y_pred is None:
                    # Try to find prediction column
                    pred_cols = [col for col in forecast.columns if 'yhat' in col]
                    if pred_cols:
                        y_pred = forecast[pred_cols[0]].values
                    else:
                        st.error("No prediction columns found in forecast")
                        y_pred = np.zeros(len(test_df))
                
                y_true = test_df['y'].values
                
                # Extract confidence intervals - handle different naming patterns
                lower_col = None
                upper_col = None
                
                # Try different possible column names
                possible_lower_names = [
                    f'yhat1 {quantile_list[0]*100:.1f}%',  # yhat1 5.0%
                    'lower1',
                    f'lower {quantile_list[0]*100:.1f}%'
                ]
                
                possible_upper_names = [
                    f'yhat1 {quantile_list[1]*100:.1f}%',  # yhat1 95.0%
                    'upper1',
                    f'upper {quantile_list[1]*100:.1f}%'
                ]
                
                for name in possible_lower_names:
                    if name in forecast.columns:
                        lower_col = name
                        break
                        
                for name in possible_upper_names:
                    if name in forecast.columns:
                        upper_col = name
                        break
                
                if lower_col and upper_col:
                    lower_bound = forecast[lower_col].values
                    upper_bound = forecast[upper_col].values
                    
                    # Clip negative values at 0 (GHI can't be negative)
                    lower_bound = np.maximum(lower_bound, 0)
                    
                else:
                    st.warning("Confidence interval columns not found. Showing predictions only.")
                    lower_bound = None
                    upper_bound = None
                
                # Calculate metrics - skip NaN values
                valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
                
                if np.sum(valid_mask) > 0:
                    y_pred_valid = y_pred[valid_mask]
                    y_true_valid = y_true[valid_mask]
                    
                    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
                    rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid)**2))
                else:
                    st.warning("No valid predictions for metrics calculation")
                    mae = 0
                    rmse = 0
                
                total_time = time.perf_counter() - start
                
                # Store results in session state
                st.session_state.neural_prophet_model = m2
                st.session_state.neural_prophet_forecast = forecast
                st.session_state.neural_prophet_results = {
                    'y_pred': y_pred,
                    'y_true': y_true,
                    'dates': test_df['ds'].values,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'mae': mae,
                    'rmse': rmse,
                    'train_time': train_time,
                    'total_time': total_time,
                    'forecast_df': forecast,
                    'valid_mask': valid_mask
                }
            
            st.success("✅ Model training complete!")
        
        # If model is trained, show results and allow date selection
        if st.session_state.neural_prophet_model is not None:
            results = st.session_state.neural_prophet_results
            
            # Performance metrics
            st.subheader("📊 Model Performance (Full Test Set)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{results['mae']:.2f} W/m²")
            
            with col2:
                st.metric("RMSE", f"{results['rmse']:.2f} W/m²")
            
            with col3:
                st.metric("Training Time", f"{results['train_time']:.2f}s")
            
            # Date selection for viewing with calendar
            st.subheader("📅 Select Forecast Period")
            
            # Get available date range from test data
            dates = pd.to_datetime(results['dates'])
            min_date = dates.min().date()
            max_date = dates.max().date()
            
            # Use date_input for calendar selection
            selected_start = st.date_input(
                "Select start date for 14-day forecast view",
                value=min_date,
                min_value=min_date,
                max_value=max_date - pd.Timedelta(days=13),
                key="forecast_start"
            )
            
            # Calculate end date (14 days later)
            selected_end = selected_start + pd.Timedelta(days=13)
            
            # Convert to datetime for filtering
            selected_start_dt = pd.Timestamp(selected_start)
            selected_end_dt = pd.Timestamp(selected_end)
            
            # Filter data for selected 14-day period
            mask = (dates >= selected_start_dt) & (dates <= selected_end_dt)
            
            filtered_dates = results['dates'][mask]
            filtered_actual = results['y_true'][mask]
            filtered_pred = results['y_pred'][mask]
            
            if len(filtered_dates) > 0:
                # Prepare dataframe for plotting
                plot_df = pd.DataFrame({
                    'Date': filtered_dates,
                    'Actual': filtered_actual,
                    'Predicted': filtered_pred,
                })
                
                # Add confidence intervals if available
                if results['lower_bound'] is not None and results['upper_bound'] is not None:
                    plot_df['Lower'] = results['lower_bound'][mask]
                    plot_df['Upper'] = results['upper_bound'][mask]
                
                # Weather context for forecast period
                st.subheader("🌤️ Weather Context for Forecast Period")
                
                # Get original weather data for this period
                weather_mask = (df['datetime'] >= selected_start_dt) & (df['datetime'] <= selected_end_dt)
                weather_df = df[weather_mask]
                
                if len(weather_df) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_temp = weather_df['temperature'].mean()
                        st.metric("Avg Temperature", f"{avg_temp:.1f} °C")
                    
                    with col2:
                        avg_humidity = weather_df['relative_humidity'].mean()
                        st.metric("Avg Humidity", f"{avg_humidity:.1f}%")
                    
                    with col3:
                        total_rain = weather_df['precipitation'].sum()
                        st.metric("Total Rainfall", f"{total_rain:.1f} mm")
                    
                    with col4:
                        avg_wind = weather_df['wind_speed'].mean()
                        st.metric("Avg Wind Speed", f"{avg_wind:.1f} m/s")
                
                # Interactive forecast plot with confidence intervals
                st.subheader(f"📈 Forecast vs Actual: {selected_start} to {selected_end}")
                
                # Create interactive plot
                fig = go.Figure()
                
                # Add confidence interval as shaded area
                if results['lower_bound'] is not None and results['upper_bound'] is not None:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([plot_df['Date'], plot_df['Date'][::-1]]),
                        y=np.concatenate([plot_df['Upper'], plot_df['Lower'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 100, 100, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='90% Confidence Interval',
                        showlegend=True,
                        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Range:</b> %{y:.1f} W/m²<extra></extra>'
                    ))
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    x=plot_df['Date'],
                    y=plot_df['Actual'],
                    mode='lines',
                    name='Actual GHI',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Actual:</b> %{y:.1f} W/m²<extra></extra>'
                ))
                
                # Add predicted values
                fig.add_trace(go.Scatter(
                    x=plot_df['Date'],
                    y=plot_df['Predicted'],
                    mode='lines',
                    name='Predicted GHI',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Predicted:</b> %{y:.1f} W/m²<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'GHI Forecast with 90% Confidence Intervals',
                    xaxis_title='Date & Time',
                    yaxis_title='GHI (W/m²)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    plot_bgcolor='rgba(240, 240, 240, 0.8)',
                    paper_bgcolor='rgba(255, 255, 255, 0.9)',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                # Add range selector for easier navigation
                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=3, label="3d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance for selected period
                st.subheader("📊 Forecast Performance (Selected 14 Days)")
                
                # Calculate metrics
                period_mae = np.mean(np.abs(plot_df['Actual'] - plot_df['Predicted']))
                period_rmse = np.sqrt(np.mean((plot_df['Actual'] - plot_df['Predicted'])**2))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Mean Absolute Error", 
                        f"{period_mae:.2f} W/m²",
                        delta=None,
                        help="Average prediction error for selected dates"
                    )
                
                with col2:
                    st.metric(
                        "Root Mean Square Error", 
                        f"{period_rmse:.2f} W/m²",
                        delta=None,
                        help="Standard deviation of prediction errors"
                    )
                
                # Forecast data table
                st.subheader("📋 Forecast Data Table")
                
                # Create a clean display dataframe
                display_df = pd.DataFrame()
                display_df['Date & Time'] = pd.to_datetime(plot_df['Date']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['Actual GHI (W/m²)'] = plot_df['Actual'].round(1)
                display_df['Predicted GHI (W/m²)'] = plot_df['Predicted'].round(1)
                
                if results['lower_bound'] is not None and results['upper_bound'] is not None:
                    display_df['Lower Bound (90% CI)'] = plot_df['Lower'].round(1)
                    display_df['Upper Bound (90% CI)'] = plot_df['Upper'].round(1)
                    display_df['Uncertainty Range'] = plot_df.apply(
                        lambda row: f"{row['Lower']:.0f} - {row['Upper']:.0f} W/m²", 
                        axis=1
                    )
                
                # Display the clean table
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Add explanation
                with st.expander("ℹ️ Understanding the Forecast"):
                    st.markdown("""
                    **Forecast Interpretation:**
                    - **Predicted GHI**: The expected GHI value
                    - **Lower Bound (90% CI)**: Minimum expected GHI with 90% confidence
                    - **Upper Bound (90% CI)**: Maximum expected GHI with 90% confidence
                    - **Uncertainty Range**: We are 90% confident that the actual GHI will fall within this range
                    
                    **Notes:**
                    - Lower bounds are clipped at 0 W/m² since GHI cannot be negative
                    - Forecasts are evaluated on historical test data (past performance)
                    - Weather context helps understand forecast conditions
                    """)
                
                # Download button for selected data
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast Data (CSV)",
                    data=csv,
                    file_name=f"neuralprophet_forecast_{selected_start}_to_{selected_end}.csv",
                    mime="text/csv",
                )
                
            else:
                st.warning(f"No forecast data available for selected date range: {selected_start} to {selected_end}")
                st.info(f"Available forecast date range: {min_date} to {max_date}")
            
        else:
            st.info("👈 Click the button above to train the Neural Prophet model with 90% confidence intervals.")
