import pandas as pd
import numpy as np
import seawater as sw
from hampel import hampel # User needs to ensure this is available
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model as keras_load_model
import joblib
#import argparse
import warnings
from tqdm import tqdm

warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn') # Suppresses sklearn version warnings

def check_data_temperature(data_df):
    """Validates presence of essential columns for temperature processing."""
    # Core columns for processing and seawater functions
    core_required = ['Prof_no', 'Temp_[°C]', 'Depth_[m]', 'Latitude_[deg]'] 
    # Note: Depth can be calculated from Pressure if Lat is present,
    # but having Depth directly is simpler for gradient calculations.
    # If only Pressure is available, process_data will attempt conversion.
    
    # Check for presence of either Depth or Pressure (with Latitude for conversion)
    has_depth = 'Depth_[m]' in data_df.columns
    has_pressure = 'Pressure_[dbar]' in data_df.columns
    has_lat = 'Latitude_[deg]' in data_df.columns

    if not ('Temp_[°C]' in data_df.columns and 'Prof_no' in data_df.columns):
        print(f"Error: Core columns 'Prof_no' or 'Temp_[°C]' missing.")
        return 2 # Critical failure

    if not (has_depth or (has_pressure and has_lat)):
        print(f"Error: Depth information missing. Need 'Depth_[m]' or ('Pressure_[dbar]' and 'Latitude_[deg]').")
        return 3 # Critical failure for gradient calculations

    # Columns expected from input CSV for ML
    ml_input_features = ['year', 'month', 'Longitude_[deg]', 'Latitude_[deg]']
    if not all(col in data_df.columns for col in ml_input_features):
        missing_ml_cols = [col for col in ml_input_features if col not in data_df.columns]
        print(f"Warning: ML input features missing from CSV: {missing_ml_cols}. ML prediction might fail.")
        # This is a warning; script proceeds, but ML step will likely error out if these are truly needed by the model.
    return 0


def process_data_temperature(data_input):
    """Applies traditional QC tests to temperature data and adds 'QF_trad' and gradient columns."""
    data = data_input.copy()

    # Ensure Depth and Pressure are available, converting if necessary
    if 'Latitude_[deg]' not in data.columns:
        print("Critical Warning: 'Latitude_[deg]' column not found. Depth/Pressure conversions and some QC might fail or be inaccurate.")
    else: # Only attempt conversion if latitude is present
        if 'Depth_[m]' not in data.columns and 'Pressure_[dbar]' in data.columns:
            try: 
                data['Depth_[m]'] = sw.eos80.dpth(data['Pressure_[dbar]'], data['Latitude_[deg]'])
                print("Info: Calculated 'Depth_[m]' from 'Pressure_[dbar]'.")
            except Exception as e: print(f"Warning: Could not calculate Depth from Pressure: {e}. Depth-dependent QC may fail.")
        elif 'Pressure_[dbar]' not in data.columns and 'Depth_[m]' in data.columns:
            try: 
                data['Pressure_[dbar]'] = sw.eos80.pres(data['Depth_[m]'], data['Latitude_[deg]'])
                print("Info: Calculated 'Pressure_[dbar]' from 'Depth_[m]'.")
            except Exception as e: print(f"Warning: Could not calculate Pressure from Depth: {e}")

    # Ensure essential 'Depth_[m]' column exists after potential conversion for gradient calcs
    if 'Depth_[m]' not in data.columns:
        print("Critical Error: 'Depth_[m]' column is not available and could not be calculated. Cannot proceed with gradient-based QC.")
        # Add empty gradient columns to prevent KeyErrors later, though ML will fail.
        data['gradientT_D'] = np.nan
        data['gradientD_T'] = np.nan
        data['QF_trad'] = 0 # Initialize QF
        return data # Return early as critical info is missing

    # Drop rows where essential columns for processing (Temp, Depth) might be NaN after all conversions
    # Prof_no should ideally not be NaN
    essential_cols_for_dropna = ['Temp_[°C]', 'Depth_[m]', 'Prof_no']
    nan_rows_before = data.isnull().any(axis=1).sum()
    if nan_rows_before > 0:
        data.dropna(subset=[col for col in essential_cols_for_dropna if col in data.columns], inplace=True)
        print(f"Info: Dropped {nan_rows_before - data.isnull().any(axis=1).sum()} rows with NaNs in essential columns.")
    
    if data.empty:
        print("Warning: DataFrame became empty after dropping NaNs. No data to process.")
        data['gradientT_D'] = np.nan # Ensure columns exist for consistency if called by main
        data['gradientD_T'] = np.nan
        data['QF_trad'] = 0
        return data
             
    # --- Nested QC Helper Functions ---
    def _bottom_top_temp_outliers(Data_bt):
        """Identifies outliers at the top and bottom of temperature profiles."""
        temp_bot_top_outlier_indices=[]
        for profile_number in Data_bt.Prof_no.unique():
            profile = Data_bt[Data_bt.Prof_no == profile_number]
            Depth_bt_vals = profile['Depth_[m]'].values
            Temp_bt_vals = profile['Temp_[°C]'].values
            current_top_outlier_indices, current_bottom_outlier_indices = [], []
            
            nanz = np.count_nonzero(np.isnan(Temp_bt_vals))
            if not (len(np.unique(Temp_bt_vals[~np.isnan(Temp_bt_vals)]))>1 and nanz != len(Temp_bt_vals)):
                continue

            h=0
            while h < len(Temp_bt_vals) and np.isnan(Temp_bt_vals[h]): h += 1
            if h < len(Temp_bt_vals): 
                starten, T_start = h, Temp_bt_vals[h]
                if (T_start < -2) or (T_start > 15): # Range check for Polar regions
                    h_search=starten
                    while h_search + 1 < len(Temp_bt_vals) and \
                          (T_start - 0.75 <= Temp_bt_vals[h_search+1] <= T_start + 0.75):
                        h_search += 1
                    current_top_outlier_indices = profile.iloc[[starten]].index.tolist()
            
            h = len(Temp_bt_vals)-1
            while h >= 0 and np.isnan(Temp_bt_vals[h]): h -= 1
            if h >= 0: 
                enden, T_end = h, Temp_bt_vals[h]
                if (T_end < -2) or (T_end > 15): # Range check for Polar regions
                    h_search=enden
                    while h_search -1 >=0 and \
                          (T_end - 0.75 <= Temp_bt_vals[h_search-1] <= T_end + 0.75):
                        h_search -= 1
                        if h_search == 0 and len(Temp_bt_vals) > 1 : break 
                    current_bottom_outlier_indices = profile.iloc[[h_search]].index.tolist()
            
            if current_top_outlier_indices: temp_bot_top_outlier_indices.extend(current_top_outlier_indices)
            if current_bottom_outlier_indices: temp_bot_top_outlier_indices.extend(current_bottom_outlier_indices)
        return list(set(temp_bot_top_outlier_indices))

    def _calculate_gradient_d_t(Data_grad_dt): # dD/dT
        """Calculates depth gradient with respect to temperature (dD/dT)."""
        unique_profil=Data_grad_dt['Prof_no'].unique()
        all_profiles_grads=[] # Changed from d_grad_list to avoid confusion
        for prof_num in tqdm(unique_profil, desc="Calc D_T Grad (dD/dT)", leave=False, disable=True):
            profil=Data_grad_dt[Data_grad_dt["Prof_no"] == prof_num].reset_index(drop=True)
            t_vals=profil['Temp_[°C]'].values
            d_vals=profil['Depth_[m]'].values
            grad_prof=[-999.0] * len(t_vals) # Initialize for all points in profile
            if len(d_vals) > 1: # Need at least two points to calculate a gradient
                grad_prof[0] = -999.0 # First point has no prior
                for i in range(len(d_vals) - 1):
                    grad_val = -999.0
                    if not(any(np.isnan([t_vals[i], t_vals[i+1], d_vals[i], d_vals[i+1]])) or \
                           any(val == -999 for val in [t_vals[i], t_vals[i+1], d_vals[i], d_vals[i+1]])):
                        delta_t = t_vals[i+1] - t_vals[i]
                        delta_d = d_vals[i+1] - d_vals[i]
                        if delta_t !=0: grad_val = delta_d / delta_t
                    grad_prof[i+1] = grad_val # Assign gradient to the second point of the pair
            elif len(d_vals) == 1:
                grad_prof[0] = -999.0
            all_profiles_grads.extend(grad_prof) # Use extend
        return np.array(all_profiles_grads)

    def _calculate_gradient_t_d(Data_grad_td): # dT/dD
        """Calculates temperature gradient with respect to depth (dT/dD)."""
        unique_profil=Data_grad_td['Prof_no'].unique()
        all_profiles_grads=[] # Changed from d_grad_list
        for prof_num in tqdm(unique_profil, desc="Calc T_D Grad (dT/dD)", leave=False, disable=True):
            profil=Data_grad_td[Data_grad_td["Prof_no"] == prof_num].reset_index(drop=True)
            t_vals=profil['Temp_[°C]'].values
            d_vals=profil['Depth_[m]'].values
            grad_prof=[-999.0] * len(t_vals) # Initialize for all points
            if len(d_vals) > 1:
                grad_prof[0] = -999.0 # First point
                for i in range(len(d_vals) - 1):
                    grad_val = -999.0
                    if not(any(np.isnan([t_vals[i], t_vals[i+1], d_vals[i], d_vals[i+1]])) or \
                           any(val == -999 for val in [t_vals[i], t_vals[i+1], d_vals[i], d_vals[i+1]])):
                        delta_t = t_vals[i+1] - t_vals[i]
                        delta_d = d_vals[i+1] - d_vals[i]
                        if delta_d !=0: grad_val = delta_t / delta_d
                    grad_prof[i+1] = grad_val
            elif len(d_vals) == 1:
                grad_prof[0] = -999.0
            all_profiles_grads.extend(grad_prof)
        return np.array(all_profiles_grads)

    def _traditional_outlier_detection_temp(Data_trad):
        """Flags data with high temperature gradient (dT/dD) in shallow depths."""
        Data_trad_copy = Data_trad.copy() # Work on a copy
        # This function focuses on dT/dD for its specific traditional QF.
        # gradientT_D (dT/dD) should be available on Data_trad_copy from the main process_data flow.
        if 'gradientT_D' not in Data_trad_copy.columns:
            print("Warning: 'gradientT_D' not found in _traditional_outlier_detection_temp. Calculating it.")
            grad_t_d_vals = _calculate_gradient_t_d(Data_trad_copy.copy())
            if len(grad_t_d_vals) == len(Data_trad_copy):
                Data_trad_copy['gradientT_D'] = grad_t_d_vals
            else:
                 Data_trad_copy['gradientT_D'] = np.nan # Fallback
        
        qf_trad_series = pd.Series(0, index=Data_trad_copy.index, name='QF_trad_from_outlier_det')
        if 'gradientT_D' in Data_trad_copy and not Data_trad_copy['gradientT_D'].isnull().all():
             condition = (Data_trad_copy['gradientT_D'] >= 0.5) & (Data_trad_copy['Depth_[m]'] <= 100) & (Data_trad_copy['gradientT_D'] != -999)
             qf_trad_series[condition] = 4
        
        # This function originally returned 'gradientD_T' as well, but it's better handled at the main processing level.
        # Here, we just return the QF flags from this specific test.
        return qf_trad_series.to_frame(name='QF_trad_val')


    def _small_temp_outliers_below_mixed_layer(Data_small_t):
        """Detects small temperature spikes below the mixed layer using Hampel filter."""
        try: from hampel import hampel
        except ImportError:
            print("Warning: 'hampel' library not found. Skipping small temp outlier detection.")
            def hampel(series, window_size, n_sigma, imputation=False): # Mock hampel, note n_sigma
                return series, pd.Series(False, index=series.index), pd.Series(dtype=float), pd.Series(dtype=float)
            
        spike_indices = []
        for profile_number in tqdm(Data_small_t.Prof_no.unique(), desc="Small Temp Spikes", leave=False, disable=True):
            profile = Data_small_t[Data_small_t.Prof_no == profile_number].copy() # Work on copy of profile
            Depth_small_t_vals = profile['Depth_[m]'].values
            Temp_small_t_vals = profile['Temp_[°C]'].values
            
            temp_gradient_vals = np.array([])
            if len(Temp_small_t_vals) > 1:
                temp_gradient_vals = np.concatenate([[np.nan],np.diff(Temp_small_t_vals)])
            elif len(Temp_small_t_vals) == 1: temp_gradient_vals = np.array([np.nan])
            else: continue # Skip if no temp values

            if not (len(Temp_small_t_vals)>7 and Depth_small_t_vals.size > 0 and Depth_small_t_vals[-1]>100): continue

            find_gradient = temp_gradient_vals.copy()
            find_gradient[Depth_small_t_vals < 100] = np.nan # Ignore mixed layer

            series_for_hampel = pd.Series(find_gradient)
            if series_for_hampel.isnull().all(): continue
            valid_gradients_indices = series_for_hampel.dropna().index
            if len(valid_gradients_indices) < 3: continue

            windowWidth = round(len(valid_gradients_indices)/10) # Adjusted window logic
            windowWidth = max(1, windowWidth + (windowWidth % 2 == 0)) 
            if windowWidth > len(valid_gradients_indices): windowWidth = max(1, len(valid_gradients_indices) - (1 if len(valid_gradients_indices)%2==0 else 0) )
            if windowWidth % 2 == 0 and windowWidth > 1: windowWidth -=1
            if windowWidth < 1 : windowWidth = 1
            
            filtered_gradients_series = series_for_hampel.copy()
            outlier_indices_hampel = pd.Series(False, index=series_for_hampel.index)

            if not series_for_hampel.isnull().all() and windowWidth < len(series_for_hampel[valid_gradients_indices]) and windowWidth > 0:
                try: 
                    # Pass n_sigma instead of n; use imputation=False to get outlier indices
                    _, temp_outlier_indices, _, _ = hampel(series_for_hampel.dropna(), window_size=windowWidth, n_sigma=6, imputation=False)
                    outlier_indices_hampel.loc[series_for_hampel.dropna().index] = temp_outlier_indices
                    # For sd_error calculation, use gradients NOT flagged by Hampel
                    filtered_gradients_series = series_for_hampel.mask(outlier_indices_hampel)
                except Exception as e_hampel: 
                    print(f"Warning: Hampel filter failed for temp profile {profile_number}: {e_hampel}.")

            sd_error = np.nanstd(filtered_gradients_series.values) # Use the series where outliers might be masked
            sd_error = sd_error if not (np.isnan(sd_error) or sd_error == 0) else 0.01
            small_threshold = sd_error # Original logic: threshold is 1*sd_error for temp
            
            # Find points where gradient exceeds threshold and is also somewhat large in magnitude
            small_indices_in_profile = np.nonzero(
                ((find_gradient > small_threshold) | (find_gradient < -small_threshold)) & 
                (abs(find_gradient) >= 0.2) # Minimum absolute gradient magnitude
            )[0]
            
            current_tips = []
            h_loop=0
            while h_loop < len(small_indices_in_profile)-1:
                idx1 = small_indices_in_profile[h_loop]
                idx2 = small_indices_in_profile[h_loop+1]
                if not (idx1 < len(temp_gradient_vals) and idx2 < len(temp_gradient_vals)):
                    h_loop += 1; continue
                
                grad_at_idx1 = temp_gradient_vals[idx1] # Gradient value at point idx1
                grad_at_idx2 = temp_gradient_vals[idx2] # Gradient value at point idx2
                if np.isnan(grad_at_idx1) or np.isnan(grad_at_idx2):
                    h_loop += 1; continue
                
                # Check for opposing gradients of similar magnitude
                # Original logic was: ten_perc = abs(eins)/2 which is 50%
                half_grad1_abs = abs(grad_at_idx1)/2 
                spike_found_flag = False
                if (grad_at_idx1 > 0 and grad_at_idx2 < 0 and (abs(grad_at_idx1) - half_grad1_abs < abs(grad_at_idx2) < abs(grad_at_idx1) + half_grad1_abs)) or \
                   (grad_at_idx1 < 0 and grad_at_idx2 > 0 and (abs(grad_at_idx1) - half_grad1_abs < abs(grad_at_idx2) < abs(grad_at_idx1) + half_grad1_abs)):
                    spike_found_flag = True
                
                if spike_found_flag:
                    # Assuming idx1 (index in temp_gradient_vals) corresponds to the point Temp_small_t_vals[idx1] being the spike.
                    # The original code's segment check was complex. Simplified to flag the point of the first anomalous gradient.
                    if idx2 == idx1 + 1: # Check if the anomalous gradients are consecutive
                         # This implies the point Temp_small_t_vals[idx1] is the spike tip
                        current_tips.append(profile.iloc[[idx1]].index.values[0])
                        h_loop += 2; continue # Skip idx2 as it's part of this spike
                    else: # If not consecutive, just flag idx1 as potentially anomalous
                        current_tips.append(profile.iloc[[idx1]].index.values[0])
                h_loop +=1
            spike_indices.extend(current_tips)
        return list(set(spike_indices))

    def _suspect_gradient_temp(Data_tsg): 
        """Flags data with suspect depth/temperature gradients (dD/dT)."""
        Data_tsg_copy = Data_tsg.copy()
        # This function uses 'gradientD_T' (dD/dT). Ensure it's available.
        if 'gradientD_T' not in Data_tsg_copy.columns:
            print("Warning: 'gradientD_T' not found in _suspect_gradient_temp. Calculating it.")
            grad_d_t_values = _calculate_gradient_d_t(Data_tsg_copy.copy())
            if len(grad_d_t_values) == len(Data_tsg_copy):
                Data_tsg_copy['gradientD_T'] = grad_d_t_values
            else:
                Data_tsg_copy['gradientD_T'] = np.nan
        
        qf_series = pd.Series(0, index=Data_tsg_copy.index, name='QF_trad_tsg')
        if 'gradientD_T' in Data_tsg_copy.columns and not Data_tsg_copy['gradientD_T'].isnull().all():
            grad_col = Data_tsg_copy['gradientD_T']
            depth_col = Data_tsg_copy['Depth_[m]']
            cond1 = (depth_col <= 100) & (grad_col.between(-0.1, 1, inclusive='both')) & (grad_col != -999)
            cond2 = (depth_col > 100) & (grad_col.between(-0.25, 5, inclusive='both')) & (grad_col != -999)
            qf_series[cond1 | cond2] = 2
        return qf_series.to_frame() # Return as DataFrame

    def _miss_temperature_value(Data_miss_t):
        """Flags missing (-999 or NaN) temperature values."""
        qf_df = pd.DataFrame(index=Data_miss_t.index, columns=['QF_trad_miss'], data=0)
        missing_condition = (Data_miss_t['Temp_[°C]'] == -999) | (Data_miss_t['Temp_[°C]'].isnull())
        qf_df.loc[missing_condition, 'QF_trad_miss'] = 5
        return qf_df[['QF_trad_miss']]
     
    # --- Main processing flow for traditional QC ---
    data['QF_trad'] = 0 # Initialize traditional QC flag
    
    # Calculate gradientT_D (dT/dD) and add to DataFrame
    grad_t_d_all = _calculate_gradient_t_d(data.copy())
    if len(grad_t_d_all) == len(data):
        data['gradientT_D'] = grad_t_d_all  
    else:
        data['gradientT_D'] = np.nan
        print(f"Warning: Length mismatch for 'gradientT_D'. Filled with NaNs.")

    # Calculate gradientD_T (dD/dT) and add to DataFrame
    grad_d_t_all = _calculate_gradient_d_t(data.copy())
    if len(grad_d_t_all) == len(data):
        data['gradientD_T'] = grad_d_t_all   
    else:
        data['gradientD_T'] = np.nan
        print(f"Warning: Length mismatch for 'gradientD_T'. Filled with NaNs.")

    # Suspect gradient check (uses dD/dT)
    tsg_results_df = _suspect_gradient_temp(data.copy()) 
    if not tsg_results_df.empty and 'QF_trad_tsg' in tsg_results_df.columns:
        condition_to_update_qf2 = (data['QF_trad'] == 0) & (tsg_results_df['QF_trad_tsg'] == 2)
        data.loc[condition_to_update_qf2[condition_to_update_qf2].index, 'QF_trad'] = 2 # Apply QF=2
     
    bt_indices = _bottom_top_temp_outliers(data.copy()) # Pass copy
    if bt_indices: data.loc[bt_indices,'QF_trad'] = 4 # Overwrites QF=2
     
    # Traditional outlier detection (uses dT/dD)
    trad_outlier_qf_series = _traditional_outlier_detection_temp(data.copy()) 
    if not trad_outlier_qf_series.empty and 'QF_trad_val' in trad_outlier_qf_series.columns:
        # Apply QF=4 from this test, potentially overwriting previous flags
        data.loc[trad_outlier_qf_series[trad_outlier_qf_series['QF_trad_val'] == 4].index, 'QF_trad'] = 4

    small_t_indices = _small_temp_outliers_below_mixed_layer(data.copy()) # Pass copy
    if small_t_indices: data.loc[small_t_indices,'QF_trad'] = 4 # Overwrites QF=2
     
    miss_t_df_results = _miss_temperature_value(data.copy()) # Pass copy
    if not miss_t_df_results.empty and 'QF_trad_miss' in miss_t_df_results.columns:
        # QF=5 for missing values should generally take precedence
        data.loc[miss_t_df_results[miss_t_df_results['QF_trad_miss'] == 5].index, 'QF_trad'] = 5

    return data

# --- ML Prediction ---
def predict_data_temperature(data_pred, model, scaler):
    """Applies a pre-trained ML model and scaler for temperature QC using 8 specified features."""
    # These are the 8 features the temperature model was trained on
    col_names = [
        'year', 'month', 'Longitude_[deg]', 'Latitude_[deg]', 'Depth_[m]',
        'Temp_[°C]',
        'gradientT_D', # dT/dD
        'gradientD_T'  # dD/dT
    ]
    
    data_pred_temp = data_pred.copy() 
    
    # Check if all required columns for ML are present
    missing_cols = [col for col in col_names if col not in data_pred_temp.columns]
    if missing_cols:
        print(f"Error: Missing columns for ML prediction: {missing_cols}. Check input CSV & processing. Returning zero predictions.")
        return np.zeros(len(data_pred))
        
    features_df = data_pred_temp[col_names].copy() # Select the 8 features

    for col in features_df.columns: 
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    features_df = features_df.fillna(0) # Imputation: fill NaNs with 0. Match training!

    if features_df.empty: 
        print("Warning: No features to scale for ML (empty DataFrame after selection/fillna).")
        return np.zeros(len(data_pred))
    
    try:
        if features_df.shape[1] != scaler.n_features_in_:
             print(f"Error: Feature count mismatch before scaling. Scaler expects {scaler.n_features_in_}, got {features_df.shape[1]}.")
             return np.zeros(len(data_pred))
        standardized_values = scaler.transform(features_df.values)
    except Exception as e:
        print(f"Error during data scaling for ML: {e}. Returning zero predictions.")
        return np.zeros(len(data_pred))
    
    try:
        predictions = model.predict(standardized_values)
        if predictions.ndim > 1: predictions = predictions[:, 0] 
    except Exception as e:
        print(f"Error during ML model prediction: {e}. Returning zero predictions.")
        return np.zeros(len(data_pred))
        
    threshold = 0.3728889524936676 
    return (predictions > threshold).astype(int)

# --- Main Execution ---
if __name__ == "__main__":
    """Main script to load data, apply traditional and ML QC for temperature, and save results."""
    default_input_file = 'TEST_DATA.csv' # Use the same test data for now
    default_output_file = 'Temperature_Output_Salacial.csv' 
    default_model_file = 'model_temp.h5' 
    default_scaler_file = 'scaler_temp.pkl'
    
    parser = argparse.ArgumentParser(description="Apply QC to Temperature data.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default=default_input_file, help='Input CSV file path.')
    parser.add_argument('--output', default=default_output_file, help='Output CSV file path.')
    parser.add_argument('--model', default=default_model_file, help='Keras model file (.h5) for temperature.')
    parser.add_argument('--scaler', default=default_scaler_file, help='Scaler file (.pkl) for temperature.')
    args = parser.parse_args()

    print(f"Starting Temperature QC: Input='{args.input}', Output='{args.output}', Model='{args.model}', Scaler='{args.scaler}'")

    try:
        if not os.path.exists(args.model): raise FileNotFoundError(f"Temperature model file not found: {args.model}")
        model_temperature = keras_load_model(args.model)
        if not os.path.exists(args.scaler): raise FileNotFoundError(f"Temperature scaler file not found: {args.scaler}")
        scaler_temperature = joblib.load(args.scaler)
        print(f"Temperature model and scaler loaded. Scaler expects {scaler_temperature.n_features_in_} features.")
    except Exception as e:
        print(f"Error loading temperature model/scaler: {e}. Exiting.")
        exit(1)

    try:
        input_data = pd.read_csv(args.input, encoding='latin1')
    except UnicodeDecodeError:
        try: input_data = pd.read_csv(args.input, encoding='windows-1252')
        except Exception as e_inner: print(f"Error reading {args.input}: {e_inner}. Exiting."); exit(1)
    except FileNotFoundError: print(f"Input file not found: {args.input}. Exiting."); exit(1)
    except Exception as e: print(f"Error reading {args.input}: {e}. Exiting."); exit(1)
    print(f"Read {len(input_data)} rows from {args.input}.")

    original_columns = input_data.columns.tolist()
    
    check_result = check_data_temperature(input_data) 

    if check_result == 0:
        print("Temperature data check OK. Processing traditional QC...")
        # process_data_temperature will add 'gradientT_D' and 'gradientD_T'
        processed_data = process_data_temperature(input_data.copy()) 
        
        if processed_data.empty:
            print("Processing resulted in an empty DataFrame. Exiting.")
            exit()
            
        print("Traditional temperature QC finished.")
        
        processed_data['ML_QF_Temp'] = 0 # Specific ML QF column for temperature
        trad_qf_col_temp = 'QF_trad' # Name used within process_data_temperature

        if trad_qf_col_temp not in processed_data.columns:
            print(f"Critical Error: Column '{trad_qf_col_temp}' not found. ML QC may be ineffective.")
            processed_data[trad_qf_col_temp] = 0 
            
        bad_data_subset = processed_data[processed_data[trad_qf_col_temp] != 0].copy()

        if not bad_data_subset.empty:
            print(f"Applying ML model to {len(bad_data_subset)} rows flagged by traditional temperature QC...")
            ml_preds = predict_data_temperature(bad_data_subset, model_temperature, scaler_temperature)
            processed_data.loc[bad_data_subset.index, 'ML_QF_Temp'] = ml_preds * bad_data_subset[trad_qf_col_temp].astype(int)
            print("ML predictions for temperature finished for bad data.")
        else:
            print("No data flagged by traditional temperature QC. ML prediction step on bad data skipped.")
        
        # Rename the traditional QF column for final output to be specific
        processed_data.rename(columns={trad_qf_col_temp: 'Trad_QF_Temp'}, inplace=True)
        
        final_cols = original_columns[:]
        if 'Trad_QF_Temp' not in final_cols: final_cols.append('Trad_QF_Temp')
        if 'ML_QF_Temp' not in final_cols: final_cols.append('ML_QF_Temp')
        
        # Optionally add gradient columns to output if desired for inspection
        # if 'gradientT_D' not in final_cols and 'gradientT_D' in processed_data.columns: final_cols.append('gradientT_D')
        # if 'gradientD_T' not in final_cols and 'gradientD_T' in processed_data.columns: final_cols.append('gradientD_T')

        final_df = processed_data[[col for col in final_cols if col in processed_data.columns]]

        try:
            final_df.to_csv(args.output, index=False)
            print(f"Temperature processing complete. Output: {args.output}")
        except Exception as e: print(f"Error saving temperature output to {args.output}: {e}")
    else:
        print(f"Input temperature data check failed (code: {check_result}). Aborting.")
