# SalaciaML_Arctic_Salinity.py

print("--- Script SalaciaML_Arctic_Salinity.py starting ---")

# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# --- 2. Configuration & Setup ---

# Set a random seed for reproducibility
SEED = 567
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define the output directory for all artifacts
OUTPUT_DIR = 'model_output_sal'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"All outputs will be saved to the '{OUTPUT_DIR}/' directory.")

# Suppress pandas' SettingWithCopyWarning, as we handle copies explicitly
pd.options.mode.chained_assignment = None

# --- 3. Data Loading and Initial Cleaning ---

# Load the dataset
try:
    data = pd.read_csv('UDASH-SML2A-Salinity-test.csv')
    print("Dataset 'UDASH-SML2A-Salinity-test' loaded successfully.")
except FileNotFoundError:
    print("Error: 'UDASH-SML2A-Salinity-test.csv' not found.")
    print("Please ensure the data file is in the same directory.")
    exit() # Exit the script if the file doesn't exist

# Drop profiles with quality flags (QF) indicating definitively bad data (e.g., 3 or 13)
data.drop(data[(data['QF_sal'].isin([3, 13]))].index, inplace=True)

# Select only profiles that have at least one traditional flag for suspicious
# gradients or spikes (QF_trad codes 2 or 4).
all_suspect_profiles = data[data['Sal_Trad_QF'].isin([2, 4])]
data2_4 = data[data.Prof_no.isin(np.unique(all_suspect_profiles.Prof_no.unique()))]

# Binarize the target variable 'QF_sal'. We want to predict 'bad' (1) vs 'good' (0).
# Codes 2, 4, 12, 14 are considered 'bad data' flags.
data2_4['QF_sal'].replace([2, 4, 12, 14], 1, inplace=True)
data2_4['QF_sal'][data2_4['QF_sal'] != 1] = 0

# Remove any rows with placeholder values like -999
data2_4.drop(data2_4[(data2_4 < -998.0).any(axis=1)].index, inplace=True)

# Some profiles have an excessive number of flagged points, which can skew the
# model. We remove profiles with more than 20 traditional flags.
trad_counts = data2_4[data2_4['Sal_Trad_QF'].isin([2, 4])].groupby('Prof_no').size()
trad_flagged_profiles = trad_counts[trad_counts > 20].index.tolist()
data2_4_filtered = data2_4[~data2_4['Prof_no'].isin(trad_flagged_profiles)]

# Create the 'bad_data' DataFrame for training.
bad_data = data2_4_filtered[data2_4_filtered['Sal_Trad_QF'] != 0].copy()
bad_data['Sal_Trad_QF'].replace([2, 4], 1, inplace=True)

print(f"Data processed. Shape of the final dataset for training/testing: {bad_data.shape}")


# --- 4. Data Splitting ---

# Define the features to be used for training the model
features = [
    'year', 'month', 'Longitude_[deg]', 'Latitude_[deg]', 'Depth_[m]',
    'Salinity_[psu]', 'Sal_gradient_[m/psu]', 'Sal_gradient_[psu/m]'
]
target = 'QF_sal'

# Split data into training, validation, and test sets based on profile numbers
profile_ids = bad_data.Prof_no.unique()
np.random.shuffle(profile_ids)

train_no, validate_no, test_no = np.split(profile_ids, [int(len(profile_ids)*0.7), int(len(profile_ids)*0.95)])

train_data = bad_data[bad_data['Prof_no'].isin(train_no.tolist())]
validation_data = bad_data[bad_data['Prof_no'].isin(validate_no.tolist())]
test_data = bad_data[bad_data['Prof_no'].isin(test_no.tolist())]


# --- 5. Data Scaling ---
scaler = StandardScaler()

# Fit on training data and transform all sets
X_train = scaler.fit_transform(train_data[features])
y_train = train_data[target]

X_val = scaler.transform(validation_data[features])
y_val = validation_data[target]

X_test = scaler.transform(test_data[features])
y_test = test_data[target]

print(f"Data split and scaled:")
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Save the scaler object
scaler_path = os.path.join(OUTPUT_DIR, 'scaler_salinity_model.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler object saved to '{scaler_path}'")


# --- 6. Model Building and Training ---

# Define the neural network architecture
model = Sequential([
    Dense(64, kernel_initializer=initializers.glorot_normal(seed=SEED), input_dim=X_train.shape[1], activation='relu'),
    Dense(32, kernel_initializer=initializers.glorot_normal(seed=SEED), activation='relu'),
    Dense(16, kernel_initializer=initializers.glorot_normal(seed=SEED), activation='relu'),
    Dense(8, kernel_initializer=initializers.glorot_normal(seed=SEED), activation='relu'),
    Dense(1, kernel_initializer=initializers.glorot_normal(seed=SEED), activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Calculate class weights to handle imbalance
class_weights_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = {int(k): v for k, v in zip(np.unique(y_train), class_weights_values)}

# Define a checkpoint to save the best model
checkpoint_filepath = os.path.join(OUTPUT_DIR, "best_salinity_model.keras")
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=0)

# Train the model
print("\nStarting model training...")
history = model.fit(
    X_train, y_train.values,  # Convert pandas Series to NumPy array
    epochs=1500,
    batch_size=16384,
    verbose=0,
    validation_data=(X_val, y_val.values), # Convert pandas Series to NumPy array
    shuffle=True,
    callbacks=[model_checkpoint_callback],
    class_weight=class_weight
)
print("Model training complete.")


# --- 7. Model Evaluation ---

# Load the best model
best_model = load_model(checkpoint_filepath)
print(f"\nLoaded best model from '{checkpoint_filepath}' for evaluation.")

# Plot training & validation loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs - Salinity')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'model_loss_curve_sal.png'), dpi=300)
# plt.show()

# Find the best threshold from the validation set
y_pred_probs_val = best_model.predict(X_val).ravel()
fpr, tpr, thresholds = roc_curve(y_val, y_pred_probs_val)
gmeans = np.sqrt(tpr * (1 - fpr))
best_threshold = thresholds[np.argmax(gmeans)]
print(f"Best Threshold found: {best_threshold:.4f}")

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic')
plt.scatter(fpr[np.argmax(gmeans)], tpr[np.argmax(gmeans)], marker='o', color='black', s=100, zorder=5, 
            label=f'Best Threshold = {best_threshold:.4f}')
plt.title('Receiver Operating Characteristic (ROC) Curve - Salinity')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_sal.png'), dpi=300)
# plt.show()

# Evaluate on the unseen test set
print("\n--- Final Evaluation on Test Set ---")
predictions_test = best_model.predict(X_test)
predictions_test_labels = (predictions_test >= best_threshold).astype(int)

# This is the crucial step that was incorrect before.
# We add the predictions back to the original `test_data` DataFrame.
test_data['ML'] = predictions_test_labels
test_data['ML_TQF'] = test_data['ML'] #* test_data['Sal_Trad_QF']

# Create confusion matrices
cm_ml_tqf = confusion_matrix(test_data[target], test_data['ML_TQF'])
cm_trad = confusion_matrix(test_data[target], test_data['Sal_Trad_QF'])

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm_ml_tqf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix: MLP - Salinity')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(cm_trad, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Confusion Matrix: Traditional Method - Salinity')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices_sal.png'), dpi=300)
# plt.show()

print("\n--- Script Finished ---")
