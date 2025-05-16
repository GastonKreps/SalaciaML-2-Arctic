# SalaciaML_Arctic_Salinity_QC.py

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# --- Configuration & Setup ---
pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings

# Data and Model Paths (Ensure these paths are correct for your environment)
DATA_FILE_PATH = 'UDASH-SML2A-Salinity.csv'  # Path to your UDASH data file
MODEL_DIR = './model_output' # Directory to save/load models and history
MODEL_CHECKPOINT_FILE = os.path.join(MODEL_DIR, "model_checkpoint_seed.keras")
SAVED_MODEL_FILE_TEMPLATE = os.path.join(MODEL_DIR, 'salacia_qc_model_seed_{}.h5')
HISTORY_FILE_TEMPLATE = os.path.join(MODEL_DIR, 'salacia_qc_model_seed_{}_history.npy')

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Define column names for features and target
FEATURE_COLUMNS = [
    'year', 'month', 'Longitude_[deg]', 'Latitude_[deg]', 'Depth_[m]',
    'Salinity_[psu]',
    'Salinity_gradient_s_d',
    'Salinity_gradient_d_s'
]
TARGET_COLUMN_ORIGINAL = 'QF.2' # Original Quality Flag
TARGET_COLUMN_TRADITIONAL = 'QF_trad' # Traditional Quality Flag (used for initial filtering)
TARGET_COLUMN_PROCESSED = 'QF.2_processed' # Processed binary target for ML

# Seed for reproducibility
SEED = 567
np.random.seed(SEED)
tf_initializer_seed = 7 # Seed for Keras initializers

# --- 1. Data Loading and Initial Preprocessing ---
print("Step 1: Loading and Initial Preprocessing...")
try:
    dtype_mapping = {
        'Cruise': str, 'Station': str, 'Platform': str, 'Type': str,
        'yyyy-mm-ddThh:mm': str, 'Source': str, 'DOI': str,
        'WOD-Cruise-ID': str, 'Salinity_gradient_s_d': float,
        'Salinity_gradient_d_s': float
    }
    data_df = pd.read_csv(DATA_FILE_PATH, dtype=dtype_mapping)
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE_PATH}' not found. Please check the path.")
    exit()

# Remove rows with QF.2 flags 3 or 13 
data_df.drop(data_df[(data_df[TARGET_COLUMN_ORIGINAL].isin([3, 13]))].index, inplace=True)

# Identify profiles with suspect gradients or spikes based on traditional flags (QF_trad = 2 or 4)
all_suspect_gradient_spikes_df = data_df[data_df[TARGET_COLUMN_TRADITIONAL].isin([2, 4])]
profiles_with_suspect_flags = data_df[data_df.Prof_no.isin(np.unique(all_suspect_gradient_spikes_df.Prof_no.unique()))].copy()

# Process QF.2 flags for these suspect profiles:
# Mark QF.2 flags 2, 4, 12, 14 (various error types) as 1 (bad for ML context)
# Mark other QF.2 flags as 0 (good for ML context)
profiles_with_suspect_flags[TARGET_COLUMN_PROCESSED] = profiles_with_suspect_flags[TARGET_COLUMN_ORIGINAL].copy()
profiles_with_suspect_flags[TARGET_COLUMN_PROCESSED].replace([2, 4, 12, 14], 1, inplace=True)
profiles_with_suspect_flags[TARGET_COLUMN_PROCESSED][profiles_with_suspect_flags[TARGET_COLUMN_PROCESSED] != 1] = 0

# Drop rows with placeholder values (-999 or similar, which might indicate missing data)
profiles_with_suspect_flags.drop(
    profiles_with_suspect_flags[(profiles_with_suspect_flags[FEATURE_COLUMNS] < -998.0).any(axis=1)].index,
    inplace=True
)

# --- 2. Further Filtering Based on Traditional Flag Counts ---
print("Step 2: Filtering Profiles by Traditional Flag Counts...")
# Count traditional flags (2 or 4) per profile
trad_flag_counts = profiles_with_suspect_flags[
    profiles_with_suspect_flags[TARGET_COLUMN_TRADITIONAL].isin([2, 4])
].groupby('Prof_no').size()

# Identify profiles with an excessive number of traditional flags (> 20) 
profiles_with_many_trad_flags = trad_flag_counts[trad_flag_counts > 20].index.tolist()
print(f"Number of profiles with more than 20 traditional flags: {len(profiles_with_many_trad_flags)}")

# Exclude these profiles from the dataset used for training/testing
filtered_data_df = profiles_with_suspect_flags[
    ~profiles_with_suspect_flags['Prof_no'].isin(profiles_with_many_trad_flags)
]
print(f"Original number of profiles (after initial QF filter): {profiles_with_suspect_flags['Prof_no'].nunique()}")
print(f"Filtered number of profiles (after high trad-flag count filter): {filtered_data_df['Prof_no'].nunique()}")

# Focus on data points that were flagged by traditional methods (QF_trad != 0)
# This step assumes we are training the ML to refine or confirm these traditional flags
bad_data_df = filtered_data_df[filtered_data_df[TARGET_COLUMN_TRADITIONAL] != 0].copy()

# Standardize QF_trad for 'bad_data': treat flags 2 and 4 as 1 (bad)
bad_data_df[TARGET_COLUMN_TRADITIONAL].replace([2, 4], 1, inplace=True)
print("Value counts for QF_trad in 'bad_data_df' after standardization:")
print(bad_data_df[TARGET_COLUMN_TRADITIONAL].value_counts())

# --- 3. Data Splitting and Feature Scaling ---
print("Step 3: Splitting Data and Scaling Features...")
unique_profile_numbers = bad_data_df.Prof_no.unique()
np.random.shuffle(unique_profile_numbers) # Shuffle profiles for random split

# Split profile numbers for train, validation, and test sets
train_prof_no, validate_prof_no, test_prof_no = np.split(
    unique_profile_numbers,
    [int(len(unique_profile_numbers) * 0.7), int(len(unique_profile_numbers) * 0.95)]
)

# Create dataframes for each set
train_df = bad_data_df[bad_data_df['Prof_no'].isin(train_prof_no.tolist())]
validation_df = bad_data_df[bad_data_df['Prof_no'].isin(validate_prof_no.tolist())]
test_df = bad_data_df[bad_data_df['Prof_no'].isin(test_prof_no.tolist())]

# Prepare for scaling
X_train_features = train_df[FEATURE_COLUMNS].reset_index(drop=True)
X_val_features = validation_df[FEATURE_COLUMNS].reset_index(drop=True)
X_test_features = test_df[FEATURE_COLUMNS].reset_index(drop=True)

y_train = train_df[TARGET_COLUMN_PROCESSED]
y_val = validation_df[TARGET_COLUMN_PROCESSED]
y_test = test_df[TARGET_COLUMN_PROCESSED]

# Initialize and fit scaler ONLY on training data
scaler = StandardScaler()
scaler.fit(X_train_features.values)

# Apply transform to all sets
X_train_scaled = scaler.transform(X_train_features.values)
X_val_scaled = scaler.transform(X_val_features.values)
X_test_scaled = scaler.transform(X_test_features.values)

# Convert scaled arrays back to DataFrames (optional, but good for consistency)
X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_COLUMNS)
X_val = pd.DataFrame(X_val_scaled, columns=FEATURE_COLUMNS)
X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)


# --- 4. Model Building and Training ---
# This section will only run if the model file doesn't exist. Otherwise, it loads the pre-trained model.
print("Step 4: Model Building and Training (or Loading)...")
saved_model_file = SAVED_MODEL_FILE_TEMPLATE.format(SEED)
history_file = HISTORY_FILE_TEMPLATE.format(SEED)

if not os.path.exists(saved_model_file):
    print(f"No pre-trained model found at {saved_model_file}. Training a new model.")
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(64, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), input_dim=input_dim, activation='relu'))
    model.add(Dense(32, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='relu')) # Removed redundant input_dim
    model.add(Dense(16, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='relu')) # Removed redundant input_dim
    model.add(Dense(8, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='relu'))
    model.add(Dense(1, kernel_initializer=initializers.glorot_normal(seed=tf_initializer_seed), activation='sigmoid'))

    model.summary()

    # Compile and train
    epochs = 1500
    batch_size = 16384
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_CHECKPOINT_FILE, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history_obj = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, # Verbose set to 1 for progress
                        validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks_list)

    # Save model (best model is already saved by checkpoint, this saves the final state)
    model.save(saved_model_file)
    np.save(history_file, history_obj.history)
    history = history_obj.history
    print(f"Model and history saved for seed {SEED}.")
else:
    print(f"Loading pre-trained model from {saved_model_file}...")
    model = load_model(saved_model_file)
    if os.path.exists(history_file):
        history = np.load(history_file, allow_pickle=True).item()
        print("Model history loaded.")
    else:
        history = None
        print("Warning: Model history file not found. Loss plot will not be generated.")
    model.summary()


# --- 5. Model Evaluation ---
print("Step 5: Model Evaluation...")

# Plot loss if history is available
if history:
    plt.figure(figsize=(15, 5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss (Seed: {SEED})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, f'model_loss_seed_{SEED}.png'))
    plt.show()
else:
    print("Skipping loss plot as history is not available.")


# Plot ROC curve on validation data to determine the best threshold
y_pred_val_proba = model.predict(X_val)[:, 0]
fpr, tpr, thresholds = roc_curve(y_val, y_pred_val_proba)
gmeans = np.sqrt(tpr * (1 - fpr)) # Geometric Mean for finding best threshold
ix = np.argmax(gmeans)
best_threshold = thresholds[ix]
print(f"Best Threshold based on G-Mean (Validation Set): {best_threshold:.4f}")

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill (AUC = 0.5)')
plt.plot(fpr, tpr, marker='.', label=f'Validation ROC (AUC = {roc_auc_score(y_val, y_pred_val_proba):.2f})')
plt.scatter(fpr[ix], tpr[ix], marker='o', s=100, color='black', label=f'Best Threshold ({best_threshold:.2f})')
plt.title(f'ROC Curve (Validation Set - Seed: {SEED})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, f'roc_curve_validation_seed_{SEED}.png'))
plt.show()


# --- 6. Predictions on Test Set and Analysis ---
print("Step 6: Predictions on Test Set and Analysis...")
# Make predictions on the test dataset using probabilities
predictions_test_proba = model.predict(X_test)[:, 0]

# Apply the best threshold found on the validation set
predictions_test_binary = (predictions_test_proba >= best_threshold).astype(int)

# Add ML predictions to the original test_df (before scaling for feature columns)
# Ensure test_df has the original features and target for analysis
test_df_analysis = test_df.copy() # Use the unscaled test_df for adding predictions
test_df_analysis['ML_Prediction'] = predictions_test_binary

# Process QF_trad: set all values to 0 except 1 (bad) for comparison
test_df_analysis.loc[:, TARGET_COLUMN_TRADITIONAL][test_df_analysis[TARGET_COLUMN_TRADITIONAL] != 1] = 0

# Create ML_TQF column (merged ML prediction with traditional flag)
# This represents cases where both ML and traditional methods agree on a "bad" flag.
test_df_analysis['ML_TQF'] = test_df_analysis['ML_Prediction'] * test_df_analysis[TARGET_COLUMN_TRADITIONAL]

# Calculate confusion matrices
y_true_test = y_test # This is the 'QF.2_processed' for the test set

# Confusion matrix for ML_TQF vs True Flags
cm_ml_tqf = confusion_matrix(y_true_test, test_df_analysis['ML_TQF'])

# Confusion matrix for Traditional Flags vs True Flags
cm_traditional = confusion_matrix(y_true_test, test_df_analysis[TARGET_COLUMN_TRADITIONAL])

# Plot ML_TQF Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_ml_tqf, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted Good', 'Predicted Bad'],
            yticklabels=['Actual Good', 'Actual Bad'])
plt.title(f'ML + Traditional Flag (ML_TQF) Confusion Matrix (Seed: {SEED})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(MODEL_DIR, f'confusion_matrix_ml_tqf_seed_{SEED}.png'))
plt.show()

# Plot Traditional Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_traditional, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted Good', 'Predicted Bad'],
            yticklabels=['Actual Good', 'Actual Bad'])
plt.title(f'Traditional Flag Only Confusion Matrix (Seed: {SEED})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(MODEL_DIR, f'confusion_matrix_traditional_seed_{SEED}.png'))
plt.show()
