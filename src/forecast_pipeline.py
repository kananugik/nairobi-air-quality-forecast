### Importing necessary libraries ###
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress
from pathlib import Path

# ======== SETTINGUP PATHS ==========
SCRIPT_DIR = Path(__file__).parent.parent          
DATA_DIR = SCRIPT_DIR / "data"
IMAGE_DIR = SCRIPT_DIR / "images"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

file_path = DATA_DIR / "nairobi_air_quality_recent.csv"

# 1. LOAD YOUR DATA
df = pd.read_csv(file_path)

df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
df = df.rename(columns={
    'pm10': 'PM10_µg_m3',
    'pm2_5': 'PM2_5_µg_m3'
})
df = df.set_index('date')
df = df[~df.index.duplicated(keep='first')]
print(f"Data loaded: {len(df)} days ({df.index.min().date()} → {df.index.max().date()})")

# 2. ADD WEATHER DATA
print("\nDownloading Temperature & Humidity from Open-Meteo...")
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": -1.2921,
    "longitude": 36.8219,
    "start_date": "2022-08-01",
    "end_date": "2026-02-18",
    "hourly": ["temperature_2m", "relative_humidity_2m"]
}

response = requests.get(url, params=params)
weather = response.json()
weather_df = pd.DataFrame({
    'date': pd.to_datetime(weather['hourly']['time']),
    'temperature_C': weather['hourly']['temperature_2m'],
    'humidity_%': weather['hourly']['relative_humidity_2m']
}).set_index('date')
weather_daily = weather_df.resample('D').mean()

# Merging
nairobi = df.join(weather_daily, how='left')
print(f"Final merged shape: {nairobi.shape}")
print("Columns:", nairobi.columns.tolist())

# 3. AQI categorized based off of pm 2.5 values
def get_us_aqi_category(pm25):
    if pd.isna(pm25): return 'Unknown'
    if pm25 <= 12.0:   return 'Good'
    elif pm25 <= 35.4: return 'Moderate'
    elif pm25 <= 55.4: return 'Unhealthy for Sensitive Groups'
    elif pm25 <= 150.4:return 'Unhealthy'
    elif pm25 <= 250.4:return 'Very Unhealthy'
    else:              return 'Hazardous'

nairobi['AQI_Category'] = nairobi['PM2_5_µg_m3'].apply(get_us_aqi_category)
print("\nAQI Distribution:")
print(nairobi['AQI_Category'].value_counts().sort_index())

# 4. MAIN TREND PLOT (PM2.5 + PM10)
plt.figure(figsize=(16, 9))
plt.plot(nairobi.index, nairobi['PM2_5_µg_m3'], color="#FD7011", linewidth=2.0, alpha=0.9, label='Daily PM2.5')
plt.plot(nairobi.index, nairobi['PM10_µg_m3'], color="#0078E1BB", linewidth=1.2, alpha=0.65, label='Daily PM10')
rolling = nairobi['PM2_5_µg_m3'].rolling(window=30, min_periods=7).mean()
plt.plot(nairobi.index, rolling, color="#5F0000", linewidth=1.8, label='30-day Rolling Average')

# Linear trend
x = np.arange(len(nairobi))
slope, intercept, r_value, _, _ = linregress(x, nairobi['PM2_5_µg_m3'])
plt.plot(nairobi.index, intercept + slope * x, color="#FFD93D", linestyle='--', linewidth=2.1,
         label=f'Linear Trend (R² = {r_value**2:.3f})')
plt.axhline(y=15, color="#058E03", linestyle=':', linewidth=1.7, label='WHO 24-hour Guideline (15 µg/m³)')
plt.title('PM2.5 & PM10 Trends in Nairobi (2022 - 2026)', fontsize=16)
plt.ylabel('Concentration (µg/m³)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(IMAGE_DIR / "PM_trends_nrb_2022_2025.png", dpi=150, bbox_inches='tight')
plt.show()

# 5. HISTOGRAMS (PM2.5, PM10, Temp, Humidity)
plt.figure(figsize=(15, 11))

cols = ['PM2_5_µg_m3', 'PM10_µg_m3', 'temperature_C', 'humidity_%']
titles = ['PM2.5 Distribution', 'PM10 Distribution', 'Temperature Distribution', 'Humidity Distribution']
colors = ['#FF6B00', '#1E88E5', '#42A5F5', '#66BB6A']

for i, col in enumerate(cols):
    plt.subplot(2, 2, i+1)
    sns.histplot(nairobi[col], bins=40, kde=True, color=colors[i], alpha=0.7)
    plt.title(titles[i], fontsize=13)
    plt.xlabel(col.replace('_', ' '))
    plt.ylabel('Frequency')
    
    mean_val = nairobi[col].mean()
    median_val = nairobi[col].median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    plt.axvline(median_val, color='blue', linestyle='-', linewidth=2, label=f'Median = {median_val:.2f}')
    if col == 'PM2_5_µg_m3':
        plt.axvline(15, color='darkgreen', linestyle=':', linewidth=2, label='WHO 24h')
    plt.legend()
plt.tight_layout(h_pad=2.5, w_pad=2.0)
plt.savefig(IMAGE_DIR / "histograms.png", dpi=150, bbox_inches='tight')
plt.show()

# 6. CORRELATION HEATMAP
plt.figure(figsize=(10, 8))
corr_cols = ['PM2_5_µg_m3', 'PM10_µg_m3', 'carbon_monoxide','sulphur_dioxide','nitrogen_dioxide','temperature_C', 'humidity_%']
sns.heatmap(nairobi[corr_cols].corr().round(3), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# Plot 4: AQI Category Breakdown
plt.figure(figsize=(8, 6))
aqi_counts = nairobi['AQI_Category'].value_counts()
sns.barplot(x=aqi_counts.index, y=aqi_counts.values, palette='viridis')
plt.title('Number of Days by AQI Category (PM2.5)')
plt.ylabel('Number of Days')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "AQI.png", dpi=150, bbox_inches='tight')
plt.show()

#################################################################################
#### predictions!!!! 3-MODEL COMPARISON: Prophet vs Random Forest vs XGBoost ####
#################################################################################
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("\n=== 3-Model Comparison: Prophet vs RF vs XGBoost ===")

# Data Preparation
daily = nairobi[['PM2_5_µg_m3', 'temperature_C', 'humidity_%']].copy()
daily = daily.rename(columns={'PM2_5_µg_m3': 'y', 'temperature_C': 'temp', 'humidity_%': 'hum'})
daily = daily.reset_index().rename(columns={'date': 'ds'})

daily['y'] = daily['y'].clip(upper=daily['y'].quantile(0.99))

# Prophet clean dataframe
prophet_df = daily[['ds', 'y']].copy()

# Features for RF & XGBoost
lags = [1, 2, 3, 4, 5, 7, 14, 21, 30]
for lag in lags:
    daily[f'lag_{lag}'] = daily['y'].shift(lag)

windows = [3, 7, 14, 21, 30]
for w in windows:
    daily[f'roll_{w}_mean'] = daily['y'].rolling(w, min_periods=1).mean()
    daily[f'roll_{w}_std'] = daily['y'].rolling(w, min_periods=1).std()
    daily[f'roll_{w}_min'] = daily['y'].rolling(w, min_periods=1).min()
    daily[f'roll_{w}_max'] = daily['y'].rolling(w, min_periods=1).max()

daily['dow'] = daily['ds'].dt.dayofweek
daily['month'] = daily['ds'].dt.month
daily['quarter'] = daily['ds'].dt.quarter
daily['is_weekend'] = (daily['dow'] >= 5).astype(int)
daily['day_of_year'] = daily['ds'].dt.dayofyear

for k in [1, 2, 3]:
    daily[f'year_sin_{k}'] = np.sin(2 * np.pi * daily['day_of_year'] * k / 365.25)
    daily[f'year_cos_{k}'] = np.cos(2 * np.pi * daily['day_of_year'] * k / 365.25)

daily['temp_hum'] = daily['temp'] * daily['hum']
daily['temp_hum_ratio'] = daily['temp'] / (daily['hum'] + 1)

daily = daily.dropna().reset_index(drop=True)

features = [col for col in daily.columns if col not in ['ds', 'y', 'lag_1']]
X = daily[features]
y = daily['y']

# Backtest
test_size = 90
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

results = {}

# 1. PROPHET MODEL
print("Training Prophet...")
prophet_train = prophet_df.iloc[:-test_size].copy()

m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    n_changepoints=25
)
m.fit(prophet_train)

future = m.make_future_dataframe(periods=test_size, freq='D')
forecast = m.predict(future)
y_pred_prophet = forecast['yhat'].iloc[-test_size:].values

results['Prophet'] = {
    'MAE': mean_absolute_error(y_test, y_pred_prophet),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_prophet)),
    'R2': r2_score(y_test, y_pred_prophet)
}

# 2. RANDOM FOREST
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=800, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

results['RandomForest'] = {
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'R2': r2_score(y_test, y_pred_rf)
}

# 3. XGBOOST
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.012, max_depth=8,
                             subsample=0.78, colsample_bytree=0.78, random_state=42,
                             early_stopping_rounds=60, eval_metric='rmse')
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test)

results['XGBoost'] = {
    'MAE': mean_absolute_error(y_test, y_pred_xgb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'R2': r2_score(y_test, y_pred_xgb)
}

# ================== COMPARISON ==================
comparison = pd.DataFrame(results).T.round(4)
print("\n=== 3-MODEL COMPARISON (Last 90 Days) ===")
print(comparison)

best_model_name = comparison['R2'].idxmax()
print(f"\nBest Model by R²: {best_model_name} (R² = {comparison.loc[best_model_name, 'R2']:.4f})")

# Use best tree model for forecast
use_model = 'XGBoost' if best_model_name in ['XGBoost', 'Prophet'] else 'RandomForest'
print(f"Using {use_model} for the 6-month forecast")

# ================== FORECASTING ==================
print(f"\nGenerating 6-month forecast using {use_model}...")

months_to_predict = [
    ('2026-02-19', '2026-02-28', 'February 2026'),
    ('2026-03-01', '2026-03-31', 'March 2026'),
    ('2026-04-01', '2026-04-30', 'April 2026'),
    ('2026-05-01', '2026-05-31', 'May 2026'),
    ('2026-06-01', '2026-06-30', 'June 2026'),
    ('2026-07-01', '2026-07-31', 'July 2026')
]

all_predictions = {}
full_forecast_dates = []
full_forecast_values = []

if use_model == 'XGBoost':
    final_model = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.012, max_depth=8,
                                   subsample=0.78, colsample_bytree=0.78, random_state=42)
else:
    final_model = RandomForestRegressor(n_estimators=800, max_depth=12, random_state=42, n_jobs=-1)

final_model.fit(X, y)

recent_y = list(daily['y'].tail(90).values)
feature_df = daily[features].iloc[-1:].copy()

for start_date, end_date, month_name in months_to_predict:
    print(f"   → Predicting {month_name}")
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    predictions = []
    
    for d in future_dates:
        feature_df['dow'] = d.dayofweek
        feature_df['month'] = d.month
        feature_df['quarter'] = d.quarter
        feature_df['is_weekend'] = 1 if d.dayofweek >= 5 else 0
        feature_df['day_of_year'] = d.dayofyear
        for k in [1, 2, 3]:
            feature_df[f'year_sin_{k}'] = np.sin(2 * np.pi * d.dayofyear * k / 365.25)
            feature_df[f'year_cos_{k}'] = np.cos(2 * np.pi * d.dayofyear * k / 365.25)

        for lag in lags:
            if f'lag_{lag}' in feature_df.columns:
                feature_df[f'lag_{lag}'] = recent_y[-lag] if len(recent_y) >= lag else recent_y[0]
        for w in windows:
            vals = recent_y[-w:]
            if f'roll_{w}_mean' in feature_df.columns:
                feature_df[f'roll_{w}_mean'] = np.mean(vals)
            if f'roll_{w}_std' in feature_df.columns:
                feature_df[f'roll_{w}_std'] = np.std(vals) if len(vals) > 1 else 0
            if f'roll_{w}_min' in feature_df.columns:
                feature_df[f'roll_{w}_min'] = np.min(vals)
            if f'roll_{w}_max' in feature_df.columns:
                feature_df[f'roll_{w}_max'] = np.max(vals)

        pred = final_model.predict(feature_df)[0]
        noisy_pred = max(0, pred + np.random.normal(0, 0.92))
        predictions.append(noisy_pred)
        recent_y.append(noisy_pred)
        if len(recent_y) > 90:
            recent_y.pop(0)

    month_df = pd.DataFrame({'Date': future_dates, 'Predicted_PM2_5': predictions})
    month_df['AQI_Category'] = month_df['Predicted_PM2_5'].apply(get_us_aqi_category)
    all_predictions[month_name] = month_df
    
    full_forecast_dates.extend(future_dates)
    full_forecast_values.extend(predictions)

# Monthly Summary & Save
monthly_summaries = []
for month_name, month_df in all_predictions.items():
    avg_pm25 = month_df['Predicted_PM2_5'].mean()
    aqi_counts = month_df['AQI_Category'].value_counts()
    total_days = len(month_df)
    summary = {
        'Month': month_name,
        'Avg PM2.5': round(avg_pm25, 2),
        'Good Days': aqi_counts.get('Good', 0),
        'Moderate Days': aqi_counts.get('Moderate', 0),
        'Unhealthy Sensitive': aqi_counts.get('Unhealthy for Sensitive Groups', 0),
        'Unhealthy Days': aqi_counts.get('Unhealthy', 0),
        '% Good': round(aqi_counts.get('Good', 0)/total_days * 100, 1),
        '% Moderate': round(aqi_counts.get('Moderate', 0)/total_days * 100, 1),
        '% Unhealthy or worse': round(
            (aqi_counts.get('Unhealthy', 0) + aqi_counts.get('Unhealthy for Sensitive Groups', 0) + 
             aqi_counts.get('Very Unhealthy', 0) + aqi_counts.get('Hazardous', 0)) / total_days * 100, 1)
    }
    monthly_summaries.append(summary)

summary_table = pd.DataFrame(monthly_summaries)
print("\n=== MONTHLY PREDICTION SUMMARY (2026) ===")
print(summary_table.to_string(index=False))

# Save outputs to data/ folder (relative)
final_forecast = pd.concat(all_predictions.values(), ignore_index=True)
final_forecast.to_csv(DATA_DIR / "forecast_2026.csv", index=False)
summary_table.to_csv(DATA_DIR / "summary_2026.csv", index=False)

#  MODEL COMPARISON BAR GRAPH PLOT
plt.figure(figsize=(12, 6))
bars = plt.bar(comparison.index, comparison['R2'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Model Performance Comparison (R² Score)', fontsize=14)
plt.ylabel('R² Score')
plt.ylim(min(comparison['R2']) - 0.1, 1.2)   
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.08,
             f"{height:.4f}", ha='center', va='bottom' if height >= 0 else 'top')

plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(IMAGE_DIR / "model_performance_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ================== HISTORICAL vs FORECAST PLOT ==================
plt.figure(figsize=(16, 9))

hist = daily.tail(365)  # pm2.5 levels for the last year
plt.plot(hist['ds'], hist['y'], color='#1f77b4', linewidth=1.8, label='Historical PM2.5')

forecast_df = pd.DataFrame({'Date': full_forecast_dates, 'Predicted_PM2_5': full_forecast_values})
plt.plot(forecast_df['Date'], forecast_df['Predicted_PM2_5'], color='#d62728', linewidth=2.0, 
         label=f'Forecast ({use_model})')

# Confidence interval
lower_ci = final_forecast['Predicted_PM2_5'] - 0.92
upper_ci = final_forecast['Predicted_PM2_5'] + 0.92

plt.fill_between(final_forecast['Date'], lower_ci, upper_ci, 
                 alpha=0.25, color='#d62728', label='Confidence Interval (±1σ)')

plt.axhline(y=15, color='darkgreen', linestyle='--', linewidth=2, label='WHO 24h Guideline (15 µg/m³)')

# Vertical line separating history and forecast
plt.axvline(x=final_forecast['Date'].min(), color='gray', linestyle=':', linewidth=2,
            label='Forecast Start')

plt.title('Nairobi PM2.5 6-Month Forecast (2026)', fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('PM2.5 Concentration (µg/m³)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(IMAGE_DIR / "Feb_to_July_2026_forecast.png", dpi=150, bbox_inches='tight')
plt.show()
