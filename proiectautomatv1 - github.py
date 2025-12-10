import cdsapi
import zipfile
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

API_URL = "https://ads.atmosphere.copernicus.eu/api"
API_KEY = "" 

def setup_api_credentials(url, key):
    rc_file = os.path.expanduser('~/.cdsapirc')
    with open(rc_file, 'w') as f:
        f.write(f"url: {url}\n")
        f.write(f"key: {key}\n")
    print(f" Configurare API completă în {rc_file} ")

def download_data():
    print(" Inițializare descărcare de la Copernicus ")
    
    output_zip = "download.zip"
    extract_folder = "date_input"
    
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    client = cdsapi.Client()

    dataset = "cams-global-reanalysis-eac4"
    request = {
        "date": ["2024-12-31/2024-12-31"],
        "time": ["12:00"],
        "data_format": "netcdf_zip",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "particulate_matter_2.5um",
            "surface_pressure"
        ],
        "area": [48.5, 20, 43, 30] # Romania si imprejurimi
    }

    # Descarca arhiva
    client.retrieve(dataset, request, output_zip)
    
    print(" Dezarhivare date ")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        extracted_files = zip_ref.namelist()
    
    nc_file = [f for f in extracted_files if f.endswith('.nc')][0]
    full_path = os.path.join(extract_folder, nc_file)
    
    print(f" Fișier pregătit: {full_path} ")
    return full_path

def proceseaza_date(cale_fisier):
    ds = xr.open_dataset(cale_fisier, engine='netcdf4')
    df = ds.to_dataframe().reset_index()
    df.dropna(inplace=True)
    
    rename_map = {
        'u10': 'u_wind', 'v10': 'v_wind', 't2m': 'temperature',
        'sp': 'pressure', 'pm2p5': 'actual_pm25' 
    }
    for col in df.columns:
        if col in rename_map:
            df.rename(columns={col: rename_map[col]}, inplace=True)

    df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
    df['temp_celsius'] = df['temperature'] - 273.15
    df['pressure_hpa'] = df['pressure'] / 100.0
    df['actual_pm25_ug'] = df['actual_pm25'] * 1e9
    
    return df
setup_api_credentials(API_URL, API_KEY)

try:
    nc_path = download_data()
    df = proceseaza_date(nc_path)
except Exception as e:
    print(f"Eroare la descărcare/procesare: {e}")
    print("Verifică cheia API și conexiunea la internet.")
    exit()

# Machine Learning
X = df[['wind_speed', 'temp_celsius', 'pressure_hpa', 'latitude', 'longitude']]
y = df['actual_pm25_ug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Antrenare modele
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

results = pd.DataFrame({
    'Real_CAMS': y_test,
    'Pred_LinearReg': y_pred_lr,
    'Pred_RandomForest': y_pred_rf
})

results_sorted = results.sort_values(by='Real_CAMS').reset_index(drop=True)

plt.figure(figsize=(14, 7))

# A. Linia Reală
plt.plot(results_sorted['Real_CAMS'], label='Real Data (CAMS)', 
         color='black', linewidth=2, linestyle='-')

# B. Linear Regression (.)
plt.plot(results_sorted['Pred_LinearReg'], label='Linear Regression', 
         color='red', marker='.', linestyle='', markersize=8)

# C. Random Forest (:)
plt.plot(results_sorted['Pred_RandomForest'], label='Random Forest', 
         color='green', linestyle=':', linewidth=3)

plt.title('Pollution prediction - PM2.5 - Copernicus (Romania)', fontsize=16)
plt.xlabel('Samples (sorted by intensity)', fontsize=12)
plt.ylabel('PM2.5 concentration [µg/m³]', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.show()

print(f"Done! RF Error (MSE): {mean_squared_error(y_test, y_pred_rf):.4f}")