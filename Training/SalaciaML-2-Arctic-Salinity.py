# SalaciaML_Arctic_Salinity.py

print("--- Script SalaciaML_Arctic_Salinity.py starting ---")

import pandas as pd
from matplotlib import pyplot as plt
import sys
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# --- Configuration & Setup ---
pd.options.mode.chained_assignment = None

# Data and Model Paths
# Original path from notebook: f'/UDASH-SML2A-Salinity.csv'
DATA_FILE_PATH = 'UDASH-SML2A-Salinity_test.csv' # <<<< UPDATE THIS PATH if your data file is elsewhere
MODEL_DIR = './model_salinity_output'
RESULTS_FILE_PATH = os.path.join(MODEL_DIR, 'model_results_salinity.txt')

MODEL_CHECKPOINT_FILE = os.path.join(MODEL_DIR, "model_checkpoint_seed_salinity.keras")
SAVED_MODEL_FILE_TEMPLATE = os.path.join(MODEL_DIR, 'sal_qc_model_seed_{}_v2.h5')
HISTORY_FILE_TEMPLATE = os.path.join(MODEL_DIR, 'sal_qc_model_seed_{}_history_v2.npy')

print(f"DATA_FILE_PATH set to: {DATA_FILE_PATH}")
print(f"MODEL_DIR set to: {MODEL_DIR}")
print(f"RESULTS_FILE_PATH set to: {RESULTS_FILE_PATH}")

try:
    print(f"Attempting to create directory: {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Directory {MODEL_DIR} ensured.")
except Exception as e:
    print(f"Error creating directory {MODEL_DIR}: {e}")
    sys.exit()
    
# Feature columns from salinity notebook 
FEATURE_COLUMNS = [
    'year', 'month', 'Longitude_[deg]', 'Latitude_[deg]', 'Depth_[m]',
    'Salinity_[psu]',
    'Salinity_gradient_s_d',
    'Salinity_gradient_d_s'
]
TARGET_COLUMN_QF_ORIGINAL = 'QF.2' # Original Quality Flag
TARGET_COLUMN_TRADITIONAL = 'QF_trad' # Traditional Quality Flag
TARGET_COLUMN_PROCESSED_QF = 'QF.2_processed' # Binary 0/1 target

SEED = 567
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf_initializer_seed = 7
print("--- Configuration and seeds set ---")

# --- 1. Data Loading and Initial Preprocessing 
print("\nStep 1: Loading and Initial Preprocessing...")
try:
    print(f"Attempting to load data from: {DATA_FILE_PATH}")
    dtype_mapping = { # From salinity notebook
        'Cruise': str, 'Station': str, 'Platform': str, 'Type': str,
        'yyyy-mm-ddThh:mm': str, 'Source': str, 'DOI': str,
        'WOD-Cruise-ID': str, 'Salinity_gradient_s_d': float,
        'Salinity_gradient_d_s':float
    }
    data_df = pd.read_csv(DATA_FILE_PATH, dtype=dtype_mapping)
    print(f"Data loaded. Shape: {data_df.shape}")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Data file '{DATA_FILE_PATH}' not found.")
    sys.exit()
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit()

data_df.drop(data_df[(data_df[TARGET_COLUMN_QF_ORIGINAL].isin([3, 13]))].index, inplace=True)
print(f"Dropped QF.2 flags 3 or 13. Shape after drop: {data_df.shape}")

# SUSPECT GRADIENT AND SPIKES (QF_trad)
all_suspect_gradient_spikes_df = data_df[data_df[TARGET_COLUMN_TRADITIONAL].isin([2, 4])]
data2_4_df = data_df[data_df.Prof_no.isin(np.unique(all_suspect_gradient_spikes_df.Prof_no.unique()))].copy()

# Process QF.2 flags for these suspect profiles
data2_4_df[TARGET_COLUMN_PROCESSED_QF] = data2_4_df[TARGET_COLUMN_QF_ORIGINAL].copy()
# Addressing FutureWarning for chained assignment
data2_4_df.loc[:, TARGET_COLUMN_PROCESSED_QF] = data2_4_df[TARGET_COLUMN_PROCESSED_QF].replace([2, 4, 12, 14], 1)
data2_4_df.loc[data2_4_df[TARGET_COLUMN_PROCESSED_QF] != 1, TARGET_COLUMN_PROCESSED_QF] = 0

# Drop rows with placeholder values 
data2_4_df.drop(data2_4_df[(data2_4_df[FEATURE_COLUMNS] < -998.0).any(axis=1)].index, inplace=True)
print(f"Shape of data2_4_df (after placeholder drop & QF processed): {data2_4_df.shape}")
print("--- Data loading and initial preprocessing finished ---")

# --- 2. Further Filtering 
print("\nStep 2: Filtering Profiles by Traditional Flag Counts...")
trad_flag_counts = data2_4_df[data2_4_df[TARGET_COLUMN_TRADITIONAL].isin([2, 4])].groupby('Prof_no').size()
profiles_with_many_trad_flags = trad_flag_counts[trad_flag_counts > 20].index.tolist()
print(f"Number of profiles with > 20 traditional flags: {len(profiles_with_many_trad_flags)}")

data2_4_filtered_df = data2_4_df[~data2_4_df['Prof_no'].isin(profiles_with_many_trad_flags)]
print(f"Profiles in data2_4_df: {data2_4_df['Prof_no'].nunique()}, Filtered profiles: {data2_4_filtered_df['Prof_no'].nunique()}")

bad_data_df = data2_4_filtered_df[data2_4_filtered_df[TARGET_COLUMN_TRADITIONAL] != 0].copy()
# Addressing FutureWarning
bad_data_df.loc[:, TARGET_COLUMN_TRADITIONAL] = bad_data_df[TARGET_COLUMN_TRADITIONAL].replace([2, 4], 1)
print("Value counts for QF_trad in 'bad_data_df':")
print(bad_data_df[TARGET_COLUMN_TRADITIONAL].value_counts())
print(f"Shape of bad_data_df: {bad_data_df.shape}")
print("--- Filtering by traditional flag counts finished ---")

# --- 3. Data Splitting and Feature Scaling 
print("\nStep 3: Splitting Data and Scaling Features...")
unique_profile_numbers = bad_data_df.Prof_no.unique()
np.random.seed(SEED) # Ensure consistent shuffle
np.random.shuffle(unique_profile_numbers)
train_prof_no, validate_prof_no, test_prof_no = np.split(
    unique_profile_numbers,
    [int(len(unique_profile_numbers) * 0.7), int(len(unique_profile_numbers) * 0.95)]
)

train_df = bad_data_df[bad_data_df['Prof_no'].isin(train_prof_no.tolist())]
validation_df = bad_data_df[bad_data_df['Prof_no'].isin(validate_prof_no.tolist())]
test_df = bad_data_df[bad_data_df['Prof_no'].isin(test_prof_no.tolist())]

X_train_features = train_df[FEATURE_COLUMNS].reset_index(drop=True)
X_val_features = validation_df[FEATURE_COLUMNS].reset_index(drop=True)
X_test_features = test_df[FEATURE_COLUMNS].reset_index(drop=True)

# Convert y_train, y_val, y_test to NumPy arrays
y_train = train_df[TARGET_COLUMN_PROCESSED_QF].values
y_val = validation_df[TARGET_COLUMN_PROCESSED_QF].values
y_test = test_df[TARGET_COLUMN_PROCESSED_QF].values

scaler = StandardScaler()
scaler.fit(X_train_features.values)
X_train_scaled = scaler.transform(X_train_features.values)
X_val_scaled = scaler.transform(X_val_features.values)
X_test_scaled = scaler.transform(X_test_features.values)

X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
X_val = pd.DataFrame(X_val_scaled, columns=FEATURE_COLUMNS)
X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)
print("Features scaled.")
print("--- Data splitting and scaling finished ---")

# --- 4. Model Building and Training 
print("\nStep 4: Model Building and Training (or Loading)...")
saved_model_file = SAVED_MODEL_FILE_TEMPLATE.format(SEED)
history_file = HISTORY_FILE_TEMPLATE.format(SEED)

results_output = []
results_output.append(f"--- Model Training/Loading for Salinity - Seed: {SEED} ---\n")

if not os.path.exists(saved_model_file):
    print(f"No pre-trained model found. Training a new model: {saved_model_file}")
    results_output.append(f"Training new model: {saved_model_file}\n")
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(64, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), input_dim=input_dim, activation='relu'))
    model.add(Dense(32, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='relu'))
    model.add(Dense(16, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='relu')) 
    model.add(Dense(8, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='relu'))
    model.add(Dense(1, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='sigmoid'))
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    results_output.append("Model Summary:\n" + model_summary + "\n")
    print(model_summary)

    epochs = 1500
    batch_size = 16384
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Salinity notebook did NOT use class_weight
    print("Class weights not used for salinity model training (as per original notebook).")
    results_output.append("Class weights not used for salinity model training.\n")

    checkpoint = ModelCheckpoint(MODEL_CHECKPOINT_FILE, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    print("Starting model training...")
    history_obj = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                            validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks_list) # No class_weight
    print("Model training finished.")
    
    print(f"Loading best model from checkpoint: {MODEL_CHECKPOINT_FILE}")
    model = load_model(MODEL_CHECKPOINT_FILE)
    model.save(saved_model_file)
    np.save(history_file, history_obj.history)
    history = history_obj.history
    results_output.append(f"Model and history saved for seed {SEED}.\n")
    print(f"Model and history saved for seed {SEED}.")
else:
    print(f"Loading pre-trained model from {saved_model_file}...")
    results_output.append(f"Loading pre-trained model: {saved_model_file}\n")
    model = load_model(saved_model_file)
    if os.path.exists(history_file):
        history = np.load(history_file, allow_pickle=True).item()
        print("Model history loaded.")
    else:
        history = None
        print("Warning: Model history file not found. Loss plot will not be generated.")
    
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    results_output.append("Loaded Model Summary:\n" + model_summary + "\n")
    print(model_summary)
print("--- Model building/training or loading finished ---")

# --- 5. Model Evaluation 
print("\nStep 5: Model Evaluation...")
results_output.append("\n--- Model Evaluation ---\n")
best_threshold_global = None

if history:
    plt.figure(figsize=(15, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - Salinity (Seed: {SEED})')
    plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout()
    loss_plot_path = os.path.join(MODEL_DIR, f'model_loss_{SEED}_sal.png')
    plt.savefig(loss_plot_path)
    print(f"Model loss plot saved to {loss_plot_path}")
    results_output.append(f"Model loss plot saved to: {loss_plot_path}\n")
    # plt.show()
else:
    print("Skipping loss plot as history is not available.")
    results_output.append("Loss plot skipped: history not available.\n")

print("Predicting on validation set for ROC curve...")
y_pred_val_proba = model.predict(X_val)[:, 0]
fpr, tpr, thresholds_roc = roc_curve(y_val, y_pred_val_proba)
roc_auc_val = roc_auc_score(y_val, y_pred_val_proba)
gmeans = np.sqrt(tpr * (1 - fpr))
ix = np.argmax(gmeans)
best_threshold_global = thresholds_roc[ix]
print(f"Best Threshold (Validation Set): {best_threshold_global:.4f} (G-Mean: {gmeans[ix]:.4f}, TPR: {tpr[ix]:.4f}, FPR: {fpr[ix]:.4f})")
results_output.append(f"Best Threshold (Validation Set): {best_threshold_global:.4f} (G-Mean: {gmeans[ix]:.4f}, TPR: {tpr[ix]:.4f}, FPR: {fpr[ix]:.4f})\n")
results_output.append(f"Validation Set AUC: {roc_auc_val:.4f}\n")

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label=f'Validation ROC (AUC={roc_auc_val:.2f})')
plt.scatter(fpr[ix], tpr[ix], marker='o', s=100, color='black', label=f'Best (Thresh={best_threshold_global:.2f})')
plt.title(f'ROC Curve - Salinity (Seed: {SEED})')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(); plt.grid(True); plt.tight_layout()
roc_plot_path = os.path.join(MODEL_DIR, f'roc_curve_{SEED}_sal.png')
plt.savefig(roc_plot_path)
print(f"ROC curve plot saved to {roc_plot_path}")
results_output.append(f"ROC curve plot saved to: {roc_plot_path}\n")
# plt.show()
print("--- Model evaluation finished ---")

# --- 6. Predictions on Test Set and Analysis 
print("\nStep 6: Predictions on Test Set and Analysis...")
results_output.append("\n--- Predictions on Test Set and Analysis ---\n")
if best_threshold_global is None:
    print("CRITICAL ERROR: Best threshold not determined.")
    results_output.append("CRITICAL ERROR: Best threshold not determined. Test set analysis skipped.\n")
    sys.exit()

print("Predicting on test set...")
predictions_test_proba = model.predict(X_test)[:, 0]
predictions_test_binary = (predictions_test_proba >= best_threshold_global).astype(int)

# Use the unscaled test_df for merging predictions and original flags
test_df_analysis = test_df.copy() # test_df contains original, unscaled values from bad_data_df
test_df_analysis['ML_Prediction'] = predictions_test_binary

# Ensure QF_trad is processed correctly: 1 for bad, 0 for good
# The bad_data_df (source of test_df) already had QF_trad processed this way
# but double check if test_df comes from a different source or needs reprocessing:
# test_df_analysis.loc[:, TARGET_COLUMN_TRADITIONAL] = test_df_analysis[TARGET_COLUMN_TRADITIONAL].replace([2, 4], 1)
test_df_analysis.loc[test_df_analysis[TARGET_COLUMN_TRADITIONAL] != 1, TARGET_COLUMN_TRADITIONAL] = 0

test_df_analysis['ML_TQF'] = test_df_analysis['ML_Prediction'] * test_df_analysis[TARGET_COLUMN_TRADITIONAL]

y_true_test = y_test # This is TARGET_COLUMN_PROCESSED_QF from the test set (NumPy array)

def get_metrics(y_true, y_pred, prefix=""):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2 : # Handle cases with only one class
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() if len(np.unique(y_true)) > 1 else (len(y_true) if np.unique(y_true)[0]==0 else 0, 0,0, len(y_true) if np.unique(y_true)[0]==1 else 0)

    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    metrics_str = (
        f"{prefix}TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}\n"
        f"{prefix}Accuracy: {accuracy:.4f}\n"
        f"{prefix}Precision: {precision:.4f}\n"
        f"{prefix}Recall: {recall:.4f}\n"
        f"{prefix}F1-Score: {f1:.4f}\n"
    )
    return (tn, fp, fn, tp), metrics_str

(tn_ml_tqf, fp_ml_tqf, fn_ml_tqf, tp_ml_tqf), metrics_ml_tqf_str = get_metrics(y_true_test, test_df_analysis['ML_TQF'].values, "ML_TQF ")
results_output.append(metrics_ml_tqf_str)
print(metrics_ml_tqf_str)

(tn_trad, fp_trad, fn_trad, tp_trad), metrics_trad_str = get_metrics(y_true_test, test_df_analysis[TARGET_COLUMN_TRADITIONAL].values, "Traditional ")
results_output.append(metrics_trad_str)
print(metrics_trad_str)

print("Plotting confusion matrices...")
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_true_test, test_df_analysis['ML_TQF']), annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted Good', 'Predicted Bad'], yticklabels=['Actual Good', 'Actual Bad'])
plt.title(f'ML+Trad Flag (ML_TQF) CM (Seed: {SEED}) - Salinity')
plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.tight_layout()
cm_ml_tqf_path = os.path.join(MODEL_DIR, f'confusion_matrix_ml_tqf_dynamic_{SEED}_sal.png')
plt.savefig(cm_ml_tqf_path)
print(f"ML_TQF confusion matrix plot saved to {cm_ml_tqf_path}")
results_output.append(f"ML_TQF confusion matrix plot saved to: {cm_ml_tqf_path}\n")
# plt.show()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_true_test, test_df_analysis[TARGET_COLUMN_TRADITIONAL]), annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted Good', 'Predicted Bad'], yticklabels=['Actual Good', 'Actual Bad'])
plt.title(f'Traditional Flag CM (Seed: {SEED}) - Salinity')
plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.tight_layout()
cm_trad_path = os.path.join(MODEL_DIR, f'confusion_matrix_traditional_dynamic_{SEED}_sal.png')
plt.savefig(cm_trad_path)
print(f"Traditional confusion matrix plot saved to {cm_trad_path}")
results_output.append(f"Traditional confusion matrix plot saved to: {cm_trad_path}\n")
# plt.show()
print("--- Predictions and analysis finished ---")

# Save results to text file
print(f"\nSaving results to {RESULTS_FILE_PATH}...")
try:
    with open(RESULTS_FILE_PATH, 'w') as f:
        for line in results_output:
            f.write(line)
    print("Results saved successfully.")
except Exception as e:
    print(f"Error saving results to file: {e}")

print("\n--- Script SalaciaML_Arctic_Salinity_QC_V2.py finished ---")
