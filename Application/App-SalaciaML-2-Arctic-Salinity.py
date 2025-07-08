# SalaciaML_Arctic_Salinity.py

print("--- Script App-SalaciaML-2-Arctic-Salinity.py starting ---")

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
from tqdm import tqdm
from hampel import hampel
import argparse
import joblib
from tensorflow.keras.models import load_model as keras_load_model
import os

# --- 2. Data Validation ---
def check_data_salinity(data_df):
    """Validates presence of essential columns for salinity processing."""
    core_required = ['Prof_no', 'Salinity_[psu]', 'Depth_[m]']
    if not all(col in data_df.columns for col in core_required):
        print(f"Error: Core columns for processing missing. Required: {core_required}")
        return 2

    ml_input_features = ['year', 'month', 'Longitude_[deg]', 'Latitude_[deg]']
    if not all(col in data_df.columns for col in ml_input_features):
        print(f"Warning: ML input features missing from CSV. Expected: {ml_input_features}")

    return 0

# --- 3. Traditional QC Processing ---
def process_data_salinity(data_input):
    """Applies traditional QC tests to salinity data and adds QF and gradient columns."""
    data = data_input.copy()

    # --- Nested QC Helper Functions ---
    def _bottom_top_sal_outliers(df):
        outlier_indices = []
        for pid in df['Prof_no'].unique():
            prof = df[df['Prof_no'] == pid]
            d, s = prof['Depth_[m]'].values, prof['Salinity_[psu]'].values
            if len(s[~np.isnan(s)]) < 2: continue
            
            # Top outlier
            h = 0
            while h < len(s) and np.isnan(s[h]): h += 1
            if h < len(s) and d[h] <= 30 and (s[h] < 10 or s[h] > 38) and s[h] > 0:
                outlier_indices.append(prof.index[h])

            # Bottom outlier
            h = len(s) - 1
            while h >= 0 and np.isnan(s[h]): h -= 1
            if h >= 0 and ((d[h] <= 30 and (s[h] < 10 or s[h] > 38)) or \
                           (d[h] > 30 and (s[h] < 25 or s[h] > 38))) and s[h] > 0:
                outlier_indices.append(prof.index[h])
        return list(set(outlier_indices))

    def _calculate_gradients(df):
        grads = {'Sal_gradient_[psu/m]': [], 'Sal_gradient_[m/psu]': []}
        for pid in df['Prof_no'].unique():
            prof = df[df['Prof_no'] == pid].reset_index(drop=True)
            s, d = prof['Salinity_[psu]'].values, prof['Depth_[m]'].values
            
            s_d_grad = np.full_like(s, -999.0, dtype=float)
            d_s_grad = np.full_like(s, -999.0, dtype=float)

            if len(s) > 1:
                delta_s = np.diff(s)
                delta_d = np.diff(d)
                with np.errstate(divide='ignore', invalid='ignore'):
                    s_d_grad[1:] = np.where(delta_d != 0, delta_s / delta_d, -999.0)
                    d_s_grad[1:] = np.where(delta_s != 0, delta_d / delta_s, -999.0)
            
            grads['Sal_gradient_[psu/m]'].extend(s_d_grad)
            grads['Sal_gradient_[m/psu]'].extend(d_s_grad)
        return grads

    def _traditional_outlier_detection_salinity(df):
        return df[(df['Sal_gradient_[psu/m]'] >= 0.5) & (df['Depth_[m]'] <= 100)].index

    def _small_salinity_outliers_below_mixed_layer(df):
        spike_indices = []
        for pid in tqdm(df['Prof_no'].unique(), desc="Small Sal Spikes", leave=False, disable=True):
            profile = df[df['Prof_no'] == pid].copy()
            if not (len(profile) > 7 and profile['Depth_[m]'].iloc[-1] > 100): continue
            
            grads = np.concatenate([[np.nan], np.diff(profile['Salinity_[psu]'])])
            grads[profile['Depth_[m]'] < 100] = np.nan
            
            series = pd.Series(grads).dropna()
            if len(series) < 3: continue
            
            win = max(1, round(len(series) / 12) | 1)
            if win >= len(series): win = max(1, len(series) - 2)

            try:
                _, outlier_idx, _, _ = hampel(series, window_size=win, n_sigma=6)
                sd_error = np.nanstd(series.mask(outlier_idx)) or 0.001
                threshold = 8 * sd_error
                anomalous_idx = series.index[np.abs(series) > threshold]
                
                for i in range(len(anomalous_idx) - 1):
                    idx1, idx2 = anomalous_idx[i], anomalous_idx[i+1]
                    if idx2 == idx1 + 1 and np.sign(grads[idx1]) != np.sign(grads[idx2]):
                        spike_indices.append(profile.index[idx1])
            except Exception as e:
                print(f"Warning: Hampel filter failed for profile {pid}: {e}.")
        return list(set(spike_indices))

    def _suspect_gradient_salinity(df):
        grad_col = df['Sal_gradient_[m/psu]']
        depth_col = df['Depth_[m]']
        cond = ((depth_col <= 100) & (grad_col.between(-50, 50))) | \
               ((depth_col > 100) & (grad_col.between(-20, 20)))
        return df[cond & (grad_col != -999)].index

    def _miss_salinity_value(df):
        return df[df['Salinity_[psu]'].isnull() | (df['Salinity_[psu]'] == -999)].index

    # --- Main processing flow ---
    data['Trad_QF_Salinity'] = 0
    
    gradient_data = _calculate_gradients(data)
    data['Sal_gradient_[psu/m]'] = gradient_data['Sal_gradient_[psu/m]']
    data['Sal_gradient_[m/psu]'] = gradient_data['Sal_gradient_[m/psu]']

    data.loc[_suspect_gradient_salinity(data), 'Trad_QF_Salinity'] = 2
    data.loc[_bottom_top_sal_outliers(data), 'Trad_QF_Salinity'] = 4
    data.loc[_traditional_outlier_detection_salinity(data), 'Trad_QF_Salinity'] = 4
    data.loc[_small_salinity_outliers_below_mixed_layer(data), 'Trad_QF_Salinity'] = 4
    data.loc[_miss_salinity_value(data), 'Trad_QF_Salinity'] = 5
            
    return data

# --- 4. ML Prediction ---
def predict_data_salinity(data_df, model, scaler):
    """Applies a pre-trained ML model for salinity QC."""
    features = [
        'year', 'month','Longitude_[deg]', 'Latitude_[deg]', 'Depth_[m]',
        'Salinity_[psu]', 'Sal_gradient_[m/psu]', 'Sal_gradient_[psu/m]'
    ]
    
    missing_cols = [col for col in features if col not in data_df.columns]
    if missing_cols:
        print(f"Error: Missing columns for ML prediction: {missing_cols}.")
        return np.zeros(len(data_df))
        
    features_df = data_df[features].copy()
    features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    if features_df.empty:
        return np.zeros(len(data_df))
    
    standardized_values = scaler.transform(features_df)
    predictions = model.predict(standardized_values).ravel()
    threshold = 0.10505
    return (predictions > threshold).astype(int)

# --- 5. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply QC to Salinity data.")
    parser.add_argument('--input', default='TEST_DATA.csv', help='Input CSV file path.')
    parser.add_argument('--output', default='Salinity_Output_Salacia.csv', help='Output CSV file path.')
    parser.add_argument('--model', default='salinity_model.keras', help='Keras model file (.keras).')
    parser.add_argument('--scaler', default='salinity_scaler.pkl', help='Scaler file (.pkl).')
    args = parser.parse_args()

    print(f"Starting Salinity QC: Input='{args.input}'")
    try:
        model_salinity = keras_load_model(args.model)
        scaler_salinity = joblib.load(args.scaler)
    except Exception as e:
        print(f"Error loading model/scaler: {e}. Exiting."); exit(1)

    try:
        input_data = pd.read_csv(args.input, encoding='latin1')
        # Handle duplicate QF columns by renaming them
        #if 'QF' in input_data.columns and 'QF.1' in input_data.columns:
            #input_data.rename(columns={'QF': 'QF_Temp', 'QF.': 'QF_Sal'}, inplace=True)
    except Exception as e:
        print(f"Error reading {args.input}: {e}. Exiting."); exit(1)

    original_columns = input_data.columns.tolist()
    
    if check_data_salinity(input_data) == 0:
        print("Data check OK. Processing traditional QC...")
        processed_data = process_data_salinity(input_data.copy())
        print("Traditional QC finished.")
        
        processed_data['ML_QF_Salinity'] = 0
        bad_data_subset = processed_data[processed_data['Trad_QF_Salinity'] != 0].copy()

        if not bad_data_subset.empty:
            print(f"Applying ML model to {len(bad_data_subset)} rows...")
            ml_preds = predict_data_salinity(bad_data_subset, model_salinity, scaler_salinity)
            processed_data.loc[bad_data_subset.index, 'ML_QF_Salinity'] = ml_preds * bad_data_subset['Trad_QF_Salinity'].astype(int)
            print("ML predictions finished.")
        else:
            print("No data flagged by traditional QC. ML prediction step skipped.")
        
        final_cols = original_columns + ['Trad_QF_Salinity', 'ML_QF_Salinity']
        final_df = processed_data[[col for col in final_cols if col in processed_data.columns]]
        
        try:
            final_df.to_csv(args.output, index=False)
            print(f"Processing complete. Output: {args.output}")
        except Exception as e:
            print(f"Error saving output: {e}")
    else:
        print("Input data check failed. Aborting.")
