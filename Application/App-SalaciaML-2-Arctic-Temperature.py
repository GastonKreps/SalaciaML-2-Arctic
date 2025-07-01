# SalaciaML_Arctic_Temperature.py

print("--- Script SalaciaML_Arctic_Temperature.py starting ---")

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
from hampel import hampel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model as keras_load_model
import joblib
import argparse
from tqdm import tqdm

# --- 2. Data Validation ---
def check_data_temperature(data_df):
    """Validates presence of essential columns for temperature processing."""
    core_required = ['Prof_no', 'Temperature_[degC]', 'Depth_[m]']
    
    missing_core = [col for col in core_required if col not in data_df.columns]
    
    if missing_core:
        print("--- Data Check Failed ---")
        print(f"Error: Core columns for processing missing.")
        print(f"Required: {core_required}")
        print(f"Found Columns in CSV: {data_df.columns.tolist()}")
        print(f"Missing: {missing_core}")
        print("--------------------------")
        return 2

    ml_input_features = ['year', 'month', 'Longitude_[deg]', 'Latitude_[deg]']
    if not all(col in data_df.columns for col in ml_input_features):
        print(f"Warning: ML input features missing. Expected: {ml_input_features}")
    
    return 0

# --- 3. Traditional QC Processing ---
def process_data_temperature(data_input):
    """Applies traditional QC tests to temperature data and adds QF and gradient columns."""
    data = data_input.copy()
    
    if 'Depth_[m]' not in data.columns:
        print("Critical Error: 'Depth_[m]' column is not available. Cannot proceed with gradient QC.")
        data['Temp_gradient_[degC/m]'] = np.nan
        data['Temp_gradient_[m/degC]'] = np.nan
        data['Trad_QF_Temperature'] = 0
        return data
        
    data.dropna(subset=['Temperature_[degC]', 'Depth_[m]', 'Prof_no'], inplace=True)
    if data.empty: return data

    # --- Nested QC Helper Functions ---
    def _bottom_top_temp_outliers(df):
        outlier_indices = []
        for pid in df['Prof_no'].unique():
            prof = df[df['Prof_no'] == pid]
            d, t = prof['Depth_[m]'].values, prof['Temperature_[degC]'].values
            if len(t[~np.isnan(t)]) < 2: continue
            
            h = 0
            while h < len(t) and np.isnan(t[h]): h += 1
            if h < len(t) and (t[h] < -2 or t[h] > 15):
                outlier_indices.append(prof.index[h])

            h = len(t) - 1
            while h >= 0 and np.isnan(t[h]): h -= 1
            if h >= 0 and (t[h] < -2 or t[h] > 15):
                outlier_indices.append(prof.index[h])
        return list(set(outlier_indices))
        
    def _calculate_gradients(df):
        grads = {'Temp_gradient_[degC/m]': [], 'Temp_gradient_[m/degC]': []}
        for pid in df['Prof_no'].unique():
            prof = df[df['Prof_no'] == pid].reset_index(drop=True)
            t, d = prof['Temperature_[degC]'].values, prof['Depth_[m]'].values
            
            t_d_grad = np.full_like(t, -999.0, dtype=float)
            d_t_grad = np.full_like(t, -999.0, dtype=float)

            if len(t) > 1:
                delta_t = np.diff(t)
                delta_d = np.diff(d)
                with np.errstate(divide='ignore', invalid='ignore'):
                    t_d_grad[1:] = np.where(delta_d != 0, delta_t / delta_d, -999.0)
                    d_t_grad[1:] = np.where(delta_t != 0, delta_d / delta_t, -999.0)
            
            grads['Temp_gradient_[degC/m]'].extend(t_d_grad)
            grads['Temp_gradient_[m/degC]'].extend(d_t_grad)
        return grads

    def _traditional_outlier_detection_temp(df):
        return df[(df['Temp_gradient_[degC/m]'] >= 0.5) & (df['Depth_[m]'] <= 100)].index

    def _small_temp_outliers_below_mixed_layer(df):
        spike_indices = []
        for pid in tqdm(df['Prof_no'].unique(), desc="Small Temp Spikes", leave=False, disable=True):
            profile = df[df['Prof_no'] == pid].copy()
            if not (len(profile) > 7 and profile['Depth_[m]'].iloc[-1] > 100): continue

            grads = np.concatenate([[np.nan], np.diff(profile['Temperature_[degC]'])])
            grads[profile['Depth_[m]'] < 100] = np.nan
            
            series = pd.Series(grads).dropna()
            if len(series) < 3: continue

            win = max(1, round(len(series) / 10) | 1)
            if win >= len(series): win = max(1, len(series) - 2)

            try:
                _, outlier_idx, _, _ = hampel(series, window_size=win, n_sigma=6)
                sd_error = np.nanstd(series.mask(outlier_idx)) or 0.01
                threshold = sd_error
                anomalous_idx = series.index[np.abs(series) > threshold]
                
                for i in range(len(anomalous_idx) - 1):
                    idx1, idx2 = anomalous_idx[i], anomalous_idx[i+1]
                    if idx2 == idx1 + 1 and np.sign(grads[idx1]) != np.sign(grads[idx2]):
                        spike_indices.append(profile.index[idx1])
            except Exception as e:
                print(f"Warning: Hampel filter failed for profile {pid}: {e}.")
        return list(set(spike_indices))

    def _suspect_gradient_temp(df):
        grad_col = df['Temp_gradient_[m/degC]']
        depth_col = df['Depth_[m]']
        cond = ((depth_col <= 100) & (grad_col.between(-0.1, 1))) | \
               ((depth_col > 100) & (grad_col.between(-0.25, 5)))
        return df[cond & (grad_col != -999)].index
        
    def _miss_temperature_value(df):
        return df[df['Temperature_[degC]'].isnull() | (df['Temperature_[degC]'] == -999)].index

    # --- Main processing flow ---
    data['Trad_QF_Temperature'] = 0
    
    gradient_data = _calculate_gradients(data)
    data['Temp_gradient_[degC/m]'] = gradient_data['Temp_gradient_[degC/m]']
    data['Temp_gradient_[m/degC]'] = gradient_data['Temp_gradient_[m/degC]']

    data.loc[_suspect_gradient_temp(data), 'Trad_QF_Temperature'] = 2
    data.loc[_bottom_top_temp_outliers(data), 'Trad_QF_Temperature'] = 4
    data.loc[_traditional_outlier_detection_temp(data), 'Trad_QF_Temperature'] = 4
    data.loc[_small_temp_outliers_below_mixed_layer(data), 'Trad_QF_Temperature'] = 4
    data.loc[_miss_temperature_value(data), 'Trad_QF_Temperature'] = 5
    
    return data

# --- 4. ML Prediction ---
def predict_data_temperature(data_df, model, scaler):
    """Applies a pre-trained ML model for temperature QC."""
    features = [
        'year', 'month', 'Longitude_[deg]', 'Latitude_[deg]', 'Depth_[m]',
        'Temperature_[degC]', 'Temp_gradient_[degC/m]', 'Temp_gradient_[m/degC]'
    ]
    
    missing_cols = [col for col in features if col not in data_df.columns]
    if missing_cols:
        print(f"Error: Missing columns for ML prediction: {missing_cols}.")
        return np.zeros(len(data_df))
        
    features_df = data_df[features].copy()
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    if features_df.empty:
        return np.zeros(len(data_df))
    
    standardized_values = scaler.transform(features_df.values)
    predictions = model.predict(standardized_values).ravel()
    threshold = 0.37288
    return (predictions > threshold).astype(int)

# --- 5. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply QC to Temperature data.")
    parser.add_argument('--input', default='TEST_DATA.csv', help='Input CSV file path.')
    parser.add_argument('--output', default='Temperature_Output_Salacia.csv', help='Output CSV file path.')
    parser.add_argument('--model', default='temperature_model.keras', help='Keras model file (.keras).')
    parser.add_argument('--scaler', default='temperature_scaler.pkl', help='Scaler file (.pkl).')
    args = parser.parse_args()

    print(f"Starting Temperature QC: Input='{args.input}'")
    try:
        model_temp = keras_load_model(args.model)
        scaler_temp = joblib.load(args.scaler)
    except Exception as e:
        print(f"Error loading model/scaler: {e}. Exiting."); exit(1)

    try:
        input_data = pd.read_csv(args.input, encoding='latin1')
        # Handle duplicate QF columns from the specific input format
        # Pandas auto-renames the second 'QF' to 'QF.1'
        if 'QF' in input_data.columns and 'QF.1' in input_data.columns:
            print("Info: Renaming duplicate 'QF' columns to 'QF_Temp' and 'QF_Sal'.")
            #input_data.rename(columns={'QF': 'QF_Temp', 'QF.1': 'QF_Sal'}, inplace=True)
    except Exception as e:
        print(f"Error reading {args.input}: {e}. Exiting."); exit(1)

    original_columns = input_data.columns.tolist()
    
    if check_data_temperature(input_data) == 0:
        print("Data check OK. Processing traditional QC...")
        processed_data = process_data_temperature(input_data.copy())
        print("Traditional QC finished.")
        
        if processed_data.empty:
            print("Processing resulted in an empty DataFrame. Exiting.")
            exit()
            
        processed_data['ML_QF_Temperature'] = 0
        bad_data_subset = processed_data[processed_data['Trad_QF_Temperature'] != 0].copy()

        if not bad_data_subset.empty:
            print(f"Applying ML model to {len(bad_data_subset)} rows...")
            ml_preds = predict_data_temperature(bad_data_subset, model_temp, scaler_temp)
            processed_data.loc[bad_data_subset.index, 'ML_QF_Temperature'] = ml_preds * bad_data_subset['Trad_QF_Temperature'].astype(int)
            print("ML predictions finished.")
        else:
            print("No data flagged by traditional QC. ML prediction step skipped.")
        
        final_cols = original_columns + ['Trad_QF_Temperature', 'ML_QF_Temperature']
        final_df = processed_data[[col for col in final_cols if col in processed_data.columns]]
        
        try:
            final_df.to_csv(args.output, index=False)
            print(f"Processing complete. Output: {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")
    else:
        print("Input data check failed. Aborting.")
