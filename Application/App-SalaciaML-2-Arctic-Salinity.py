import pandas as pd
import numpy as np
from tqdm import tqdm
from hampel import hampel # Ensure this is available if used
import argparse
import joblib
from tensorflow.keras.models import load_model as keras_load_model
import os

def check_data_salinity(data_df):
    """Validates presence of essential columns for salinity processing."""
    # Check for core processing columns
    core_required = ['Prof_no', 'Salinity_[psu]', 'Depth_[m]'] # Depth is needed for gradients
    if not all(col in data_df.columns for col in core_required): 
        print(f"Error: Core columns for processing missing. Required: {core_required}")
        return 2
    
    # Check for ML feature columns (some might be generated, others must be input)
    # Columns expected from input CSV for ML: 'year', 'month', 'Longitude_[deg]', 'Latitude_[deg]'
    ml_input_features = ['year', 'month', 'Longitude_[deg]', 'Latitude_[deg]']
    if not all(col in data_df.columns for col in ml_input_features):
        print(f"Warning: Some ML input features might be missing from CSV. Expected: {ml_input_features}")
        # This is a warning for now, as the script will proceed but ML might fail later if they are truly missing.
        # A stricter check could return an error here.

    # Optional columns for initial check (Pressure)
    optional_columns = ['Pressure_[dbar]']
    if not any(col in data_df.columns for col in optional_columns):
        # This was original return 3, but less critical than core or ML features
        print("Warning: Optional column 'Pressure_[dbar]' not found.")
        # return 3 # Or just a warning

    return 0

def process_data_salinity(data_input):
    """Applies traditional QC tests to salinity data and adds 'QF_trad_salinity' and gradient columns."""
    data = data_input.copy()

    # --- Nested QC Helper Functions ---
    def _bottom_top_sal_outliers(Data_bt):
        """Identifies outliers at the top and bottom of salinity profiles."""
        sal_bot_top_outlier_indices = []
        for profile_number in Data_bt['Prof_no'].unique():
            profile = Data_bt[Data_bt['Prof_no'] == profile_number]
            Depth_bt = profile['Depth_[m]'].values
            Salinity_bt = profile['Salinity_[psu]'].values
            current_top_outlier_indices, current_bottom_outlier_indices = [], []
            
            nanz = np.count_nonzero(np.isnan(Salinity_bt))
            if not (len(np.unique(Salinity_bt[~np.isnan(Salinity_bt)])) > 1 and nanz != len(Salinity_bt)):
                continue

            h = 0
            while h < len(Salinity_bt) and np.isnan(Salinity_bt[h]): h += 1
            if h < len(Salinity_bt): 
                starten, S_start, D_start = h, Salinity_bt[h], Depth_bt[h]
                if D_start <= 30 and (S_start < 10 or S_start > 38) and S_start > 0:
                    idx_pointer = starten
                    while idx_pointer + 1 < len(Salinity_bt) and \
                          (S_start - 2 <= Salinity_bt[idx_pointer+1] <= S_start + 2):
                        idx_pointer += 1
                    current_top_outlier_indices = profile.iloc[[starten]].index.tolist()

            h = len(Salinity_bt) - 1
            while h >= 0 and np.isnan(Salinity_bt[h]): h -= 1
            if h >= 0: 
                enden_idx_in_profile, S_end, D_end = h, Salinity_bt[h], Depth_bt[h]
                is_anomalous_bottom = (D_end <= 30 and (S_end < 10 or S_end > 38) and S_end > 0) or \
                                      (D_end > 30 and (S_end < 25 or S_end > 38) and S_end > 0)
                if is_anomalous_bottom:
                    h_search = enden_idx_in_profile
                    while h_search - 1 >= 0 and \
                          (S_end - 2 <= Salinity_bt[h_search-1] <= S_end + 2):
                        if D_end > 30 and Depth_bt[h_search-1] < 30: break
                        h_search -= 1
                        if h_search == 0: break
                    current_bottom_outlier_indices = profile.iloc[[h_search]].index.tolist()
            
            if current_top_outlier_indices: sal_bot_top_outlier_indices.extend(current_top_outlier_indices)
            if current_bottom_outlier_indices: sal_bot_top_outlier_indices.extend(current_bottom_outlier_indices)
        return list(set(sal_bot_top_outlier_indices))

    def _calculate_gradient_s_d(Data_grad_sd): # dS/dD
        """Calculates salinity gradient with respect to depth (dS/dD)."""
        unique_profil = Data_grad_sd['Prof_no'].unique()
        s_d_grad_list = []
        for prof_num in tqdm(unique_profil, desc="Calc S_D Grad (dS/dD)", leave=False, disable=True): 
            profil = Data_grad_sd[Data_grad_sd["Prof_no"] == prof_num].reset_index(drop=True)
            s_vals, d_vals = profil['Salinity_[psu]'].values, profil['Depth_[m]'].values
            grad = [-999.0] * len(s_vals)
            if len(d_vals) > 1:
                for i in range(len(d_vals) - 1):
                    current_grad_val = -999.0
                    if not(any(np.isnan([s_vals[i], s_vals[i+1], d_vals[i], d_vals[i+1]])) or \
                           any(val == -999 for val in [s_vals[i], s_vals[i+1], d_vals[i], d_vals[i+1]])):
                        delta_d, delta_s = d_vals[i+1] - d_vals[i], s_vals[i+1] - s_vals[i]
                        if delta_d != 0: current_grad_val = delta_s / delta_d
                    grad[i+1] = current_grad_val
            s_d_grad_list.extend(grad)
        return np.array(s_d_grad_list)

    def _traditional_outlier_detection_salinity(Data_trad):
        """Flags data with high salinity gradient (dS/dD) in shallow depths."""
        if 'Salinity_gradient_s_d' not in Data_trad.columns: # Expects specific name now
            Data_trad['Salinity_gradient_s_d'] = _calculate_gradient_s_d(Data_trad.copy())
        data1 = Data_trad.loc[(Data_trad['Salinity_gradient_s_d'] >= 0.5) & (Data_trad['Depth_[m]'] <= 100)].copy()
        if not data1.empty: data1.loc[:, 'QF_trad_salinity'] = 4
        return data1
        
    def _small_salinity_outliers_below_mixed_layer(Data_small_s):
        """Detects small salinity spikes below the mixed layer using Hampel filter."""
        try: from hampel import hampel
        except ImportError:
            print("Warning: 'hampel' library not found. Skipping small outlier detection.")
            def hampel(series, window_size, n_sigma, imputation=False): # Mock hampel, note n_sigma
                return series, pd.Series(False, index=series.index), pd.Series(dtype=float), pd.Series(dtype=float)
            
        spike_indices = []
        for profile_number in tqdm(Data_small_s['Prof_no'].unique(), desc="Small Sal Spikes", leave=False, disable=True): 
            profile = Data_small_s[Data_small_s['Prof_no'] == profile_number].copy()
            Depth_small_s, Salinity_small_s = profile['Depth_[m]'].values, profile['Salinity_[psu]'].values
            
            if not (len(Salinity_small_s) > 7 and Depth_small_s.size > 0 and Depth_small_s[-1] > 100): continue

            salinity_gradient_vals = np.concatenate([[np.nan], np.diff(Salinity_small_s)])
            find_gradient_small_s = salinity_gradient_vals.copy()
            find_gradient_small_s[Depth_small_s < 100] = np.nan

            series_for_hampel = pd.Series(find_gradient_small_s)
            if series_for_hampel.isnull().all(): continue
            valid_gradients_indices = series_for_hampel.dropna().index
            if len(valid_gradients_indices) < 3: continue

            windowWidth = round(len(valid_gradients_indices)/12)
            windowWidth = max(1, windowWidth + (windowWidth % 2 == 0)) 
            if windowWidth > len(valid_gradients_indices): windowWidth = max(1, len(valid_gradients_indices) - (1 if len(valid_gradients_indices)%2==0 else 0) )
            if windowWidth % 2 == 0 and windowWidth > 1: windowWidth -=1
            if windowWidth < 1 : windowWidth = 1

            filtered_gradients_series = series_for_hampel.copy()
            outlier_indices_hampel = pd.Series(False, index=series_for_hampel.index)
            if not series_for_hampel.isnull().all() and windowWidth < len(series_for_hampel[valid_gradients_indices]) and windowWidth > 0:
                try: 
                    # Pass n_sigma instead of n
                    _, temp_outlier_indices, _, _ = hampel(series_for_hampel.dropna(), window_size=windowWidth, n_sigma=6, imputation=False)
                    outlier_indices_hampel.loc[series_for_hampel.dropna().index] = temp_outlier_indices
                    filtered_gradients_series = series_for_hampel.mask(outlier_indices_hampel)
                except Exception as e_hampel: 
                    print(f"Warning: Hampel filter failed for profile {profile_number}: {e_hampel}.")

            sd_error = np.nanstd(filtered_gradients_series.mask(outlier_indices_hampel).values)
            sd_error = sd_error if not (np.isnan(sd_error) or sd_error == 0) else 0.001
            small_threshold = 8 * sd_error
            
            anomalous_indices_in_profile = np.nonzero(
                ((find_gradient_small_s > small_threshold) | (find_gradient_small_s < -small_threshold)) & 
                (np.abs(find_gradient_small_s) >= 0.01) )[0]

            current_spike_tips = []
            h = 0
            while h < len(anomalous_indices_in_profile) - 1:
                idx1, idx2 = anomalous_indices_in_profile[h], anomalous_indices_in_profile[h+1]
                if not(idx1 < len(salinity_gradient_vals) and idx2 < len(salinity_gradient_vals)):
                    h +=1; continue
                grad1, grad2 = salinity_gradient_vals[idx1], salinity_gradient_vals[idx2]
                if np.isnan(grad1) or np.isnan(grad2):
                    h += 1; continue
                
                half_grad1_abs = abs(grad1) / 2 
                spike_found = (grad1 > 0 and grad2 < 0 and (abs(grad1) - half_grad1_abs < abs(grad2) < abs(grad1) + half_grad1_abs)) or \
                              (grad1 < 0 and grad2 > 0 and (abs(grad1) - half_grad1_abs < abs(grad2) < abs(grad1) + half_grad1_abs))
                
                if spike_found:
                    if idx2 == idx1 + 1: 
                        current_spike_tips.append(profile.iloc[[idx1]].index.values[0])
                        h += 2; continue 
                    else: 
                         current_spike_tips.append(profile.iloc[[idx1]].index.values[0])
                h += 1
            spike_indices.extend(current_spike_tips)
        return list(set(spike_indices))

    def _calculate_gradient_d_s(Data_sg): # dD/dS
        """Calculates depth gradient with respect to salinity (dD/dS)."""
        unique_profil = Data_sg['Prof_no'].unique()
        d_grad_list = []
        for prof_num in tqdm(unique_profil, desc="Calc D_S Grad (dD/dS)", leave=False, disable=True): 
            profil = Data_sg[Data_sg["Prof_no"] == prof_num].reset_index(drop=True)
            s_vals, d_vals = profil['Salinity_[psu]'].values, profil['Depth_[m]'].values
            grad = [-999.0] * len(s_vals)
            if len(s_vals) > 1:
                for i in range(len(s_vals) - 1):
                    current_grad_val = -999.0
                    if not(any(np.isnan([s_vals[i], s_vals[i+1], d_vals[i], d_vals[i+1]])) or \
                           any(val == -999 for val in [s_vals[i], s_vals[i+1], d_vals[i], d_vals[i+1]])):
                        delta_s, delta_d = s_vals[i+1] - s_vals[i], d_vals[i+1] - d_vals[i]
                        if delta_s != 0: current_grad_val = delta_d / delta_s
                    grad[i+1] = current_grad_val
            d_grad_list.extend(grad)
        return np.array(d_grad_list)

    def _suspect_gradient_salinity(Data_sgs):
        """Flags data with suspect depth/salinity gradients (dD/dS)."""
        Data_sgs_copy = Data_sgs.copy()
        # This function uses 'Salinity_gradient_d_s'. Ensure it's calculated if not present.
        if 'Salinity_gradient_d_s' not in Data_sgs_copy.columns:
            grad_d_s_values = _calculate_gradient_d_s(Data_sgs_copy.copy()) 
            Data_sgs_copy['Salinity_gradient_d_s'] = grad_d_s_values if len(grad_d_s_values) == len(Data_sgs_copy) else np.nan
        
        Data_sgs_copy['QF_trad_salinity_sg'] = 0
        if 'Salinity_gradient_d_s' in Data_sgs_copy.columns and not Data_sgs_copy['Salinity_gradient_d_s'].isnull().all():
            grad_col = Data_sgs_copy['Salinity_gradient_d_s']
            depth_col = Data_sgs_copy['Depth_[m]']
            cond1 = (depth_col <= 100) & (grad_col.between(-50, 50, inclusive='both')) & (grad_col != -999)
            cond2 = (depth_col > 100) & (grad_col.between(-20, 20, inclusive='both')) & (grad_col != -999)
            Data_sgs_copy.loc[cond1 | cond2, 'QF_trad_salinity_sg'] = 2
        return Data_sgs_copy[['QF_trad_salinity_sg']]

    def _miss_salinity_value(Data_miss_s):
        """Flags missing (-999 or NaN) salinity values."""
        qf_df = pd.DataFrame(index=Data_miss_s.index, columns=['QF_trad_salinity_miss'], data=0)
        missing_condition = (Data_miss_s['Salinity_[psu]'] == -999) | (Data_miss_s['Salinity_[psu]'].isnull())
        qf_df.loc[missing_condition, 'QF_trad_salinity_miss'] = 5
        return qf_df[['QF_trad_salinity_miss']]

    # --- Main processing flow for traditional QC ---
    data['QF_trad_salinity'] = 0 
    
    # Calculate dS/dD and name it as expected by ML features
    grad_s_d_all = _calculate_gradient_s_d(data.copy())
    data['Salinity_gradient_s_d'] = grad_s_d_all if len(grad_s_d_all) == len(data) else np.nan
    if len(grad_s_d_all) != len(data): print(f"Warning: Length mismatch for 'Salinity_gradient_s_d'.")

    # Calculate dD/dS and name it as expected by ML features
    grad_d_s_all = _calculate_gradient_d_s(data.copy())
    data['Salinity_gradient_d_s'] = grad_d_s_all if len(grad_d_s_all) == len(data) else np.nan
    if len(grad_d_s_all) != len(data): print(f"Warning: Length mismatch for 'Salinity_gradient_d_s'.")


    sg_results = _suspect_gradient_salinity(data.copy()) # This will use 'Salinity_gradient_d_s'
    if not sg_results.empty:
        data.loc[(data['QF_trad_salinity'] == 0) & (sg_results['QF_trad_salinity_sg'] == 2), 'QF_trad_salinity'] = 2

    bt_indices = _bottom_top_sal_outliers(data.copy())
    if bt_indices: data.loc[bt_indices, 'QF_trad_salinity'] = 4
    
    # This will use 'Salinity_gradient_s_d'
    trad_outliers_df = _traditional_outlier_detection_salinity(data.copy()) 
    if not trad_outliers_df.empty:
        data.loc[trad_outliers_df.index, 'QF_trad_salinity'] = trad_outliers_df['QF_trad_salinity']

    small_spike_idx = _small_salinity_outliers_below_mixed_layer(data.copy())
    if small_spike_idx: data.loc[small_spike_idx, 'QF_trad_salinity'] = 4

    missing_sal_df = _miss_salinity_value(data.copy())
    if not missing_sal_df.empty:
        data.loc[missing_sal_df[missing_sal_df['QF_trad_salinity_miss'] == 5].index, 'QF_trad_salinity'] = 5
            
    return data

def predict_data_salinity(data_df, model, scaler):
    """Applies a pre-trained ML model for salinity QC using the 8 specified features."""
    # These are the 8 features the model was trained on
    col_names= [
        'year', 'month','Longitude_[deg]', 'Latitude_[deg]', 'Depth_[m]',
        'Salinity_[psu]', 
        'Salinity_gradient_s_d', # Calculated as dS/dD
        'Salinity_gradient_d_s'  # Calculated as dD/dS
    ]
    
    data_df_temp = data_df.copy()
    
    # Check if all required columns for ML are present
    missing_cols = [col for col in col_names if col not in data_df_temp.columns]
    if missing_cols:
        print(f"Error: Missing columns required for ML prediction: {missing_cols}. Check input CSV and processing steps. Returning zero predictions.")
        return np.zeros(len(data_df))
        
    features_df = data_df_temp[col_names].copy() # Select the 8 features

    # Ensure all feature columns are numeric and handle NaNs before scaling
    for col in features_df.columns: 
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # Imputation strategy: fill NaNs with 0. This should match how the model was trained.
    features_df = features_df.fillna(0) 
    
    if features_df.empty: 
        print("Warning: No features to scale for ML prediction (empty DataFrame).")
        return np.zeros(len(data_df))

    try:
        # Scaler expects 8 features, features_df should now have 8
        if features_df.shape[1] != scaler.n_features_in_:
             print(f"Error: Feature count mismatch before scaling. Expected {scaler.n_features_in_}, got {features_df.shape[1]}.")
             return np.zeros(len(data_df))
        standardized_values = scaler.transform(features_df.values)
    except Exception as e:
        print(f"Error during data scaling for ML: {e}. Returning zero predictions.")
        return np.zeros(len(data_df))

    try:
        predictions = model.predict(standardized_values)
        if predictions.ndim > 1: predictions = predictions[:, 0] 
    except Exception as e:
        print(f"Error during ML model prediction: {e}. Returning zero predictions.")
        return np.zeros(len(data_df))

    threshold = 0.10505027323961258 
    return (predictions > threshold).astype(int)

# --- Main Execution ---
if __name__ == "__main__":
    """Main script to load data, apply traditional and ML QC, and save results."""
    default_input_file = 'TEST_DATA.csv' 
    default_output_file = 'Salinity_Output_Salacial.csv' 
    default_model_file = 'model_sal.h5' 
    default_scaler_file = 'scaler_sal.pkl'
    
    parser = argparse.ArgumentParser(description="Apply QC to Salinity data.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default=default_input_file, help='Input CSV file path.')
    parser.add_argument('--output', default=default_output_file, help='Output CSV file path.')
    parser.add_argument('--model', default=default_model_file, help='Keras model file (.h5).')
    parser.add_argument('--scaler', default=default_scaler_file, help='Scaler file (.pkl).')
    args = parser.parse_args()

    print(f"Starting Salinity QC: Input='{args.input}', Output='{args.output}', Model='{args.model}', Scaler='{args.scaler}'")

    try:
        if not os.path.exists(args.model): raise FileNotFoundError(f"Model file not found: {args.model}")
        model_salinity = keras_load_model(args.model)
        if not os.path.exists(args.scaler): raise FileNotFoundError(f"Scaler file not found: {args.scaler}")
        scaler_salinity = joblib.load(args.scaler)
        print(f"Model and scaler loaded. Scaler expects {scaler_salinity.n_features_in_} features.")
    except Exception as e:
        print(f"Error loading model/scaler: {e}. Exiting.")
        exit(1)

    try:
        input_data_salinity = pd.read_csv(args.input, encoding='latin1')
    except UnicodeDecodeError:
        try: input_data_salinity = pd.read_csv(args.input, encoding='windows-1252')
        except Exception as e_inner: print(f"Error reading {args.input}: {e_inner}. Exiting."); exit(1)
    except FileNotFoundError: print(f"Input file not found: {args.input}. Exiting."); exit(1)
    except Exception as e: print(f"Error reading {args.input}: {e}. Exiting."); exit(1)
    print(f"Read {len(input_data_salinity)} rows from {args.input}.")

    original_columns = input_data_salinity.columns.tolist()
    
    # Perform data check, including presence of necessary ML input columns from CSV
    check_result = check_data_salinity(input_data_salinity) 

    if check_result == 0:
        print("Data check OK. Processing traditional QC...")
        # process_data_salinity will now add 'Salinity_gradient_s_d' and 'Salinity_gradient_d_s'
        processed_data = process_data_salinity(input_data_salinity.copy()) 
        print("Traditional QC finished.")
        
        processed_data['ML_QF_Salinity'] = 0 
        trad_qf_col = 'QF_trad_salinity' 

        if trad_qf_col not in processed_data.columns:
            print(f"Critical Error: Column '{trad_qf_col}' not found. ML QC may be ineffective.")
            processed_data[trad_qf_col] = 0 
            
        bad_data_subset = processed_data[processed_data[trad_qf_col] != 0].copy()

        if not bad_data_subset.empty:
            print(f"Applying ML model to {len(bad_data_subset)} rows flagged by traditional QC...")
            # predict_data_salinity now expects the 8 features
            ml_preds = predict_data_salinity(bad_data_subset, model_salinity, scaler_salinity)
            processed_data.loc[bad_data_subset.index, 'ML_QF_Salinity'] = ml_preds * bad_data_subset[trad_qf_col].astype(int)
            print("ML predictions finished for bad data.")
        else:
            print("No data flagged by traditional QC. ML prediction step on bad data skipped.")
        
        processed_data.rename(columns={trad_qf_col: 'Trad_QF_Salinity'}, inplace=True)
        
        final_cols = original_columns[:]
        # Add new QF columns if not already present from original (unlikely for these)
        if 'Trad_QF_Salinity' not in final_cols: final_cols.append('Trad_QF_Salinity')
        if 'ML_QF_Salinity' not in final_cols: final_cols.append('ML_QF_Salinity')
        
        # Add generated gradient columns to output if they were not original and user wants to see them
        # For now, just ensure the QF flags are there. User can modify if they want all 8 ML features in output.
        # Example:
        # if 'Salinity_gradient_s_d' not in final_cols: final_cols.append('Salinity_gradient_s_d')
        # if 'Salinity_gradient_d_s' not in final_cols: final_cols.append('Salinity_gradient_d_s')

        final_df = processed_data[[col for col in final_cols if col in processed_data.columns]]

        try:
            final_df.to_csv(args.output, index=False)
            print(f"Processing complete. Output: {args.output}")
        except Exception as e: print(f"Error saving output to {args.output}: {e}")
    else:
        print(f"Input data check failed (code: {check_result}). Aborting.")
