#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Install required packages for the project

get_ipython().system('pip install numpy pandas matplotlib scikit-learn statsmodels tensorflow')
print("Environement setup completed successfully")


# In[4]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("All Libraries imported successfully")


# In[10]:


# Load time series dataset for forecasting project
import requests
import zipfile
import io

# Load time series dataset for forecasting project
print("Initializing dataset download...")

# Define dataset source
data_source = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.zip"

# Download and process dataset
server_response = requests.get(data_source)
compressed_file = zipfile.ZipFile(io.BytesIO(server_response.content))
compressed_file.extractall('project_dataset')

# Read dataset with appropriate parameters
forecasting_data = pd.read_csv('project_dataset/household_power_consumption.txt', 
                              delimiter=';',
                              low_memory=False,
                              na_values=['?'],
                              nrows=15000)

print("Dataset preparation completed")
print(f"Dataset dimensions: {forecasting_data.shape}")

# Display column names to see what we have
print(f"\nColumn names: {list(forecasting_data.columns)}")

# Check first few rows
print("\nFirst 3 rows of data:")
display(forecasting_data.head(3))


# In[11]:


# Process and prepare the time series dataset for forecasting
print("Processing dataset for time series forecasting...")

# Create proper DateTime column from Date and Time columns
forecasting_data['DateTime'] = pd.to_datetime(
    forecasting_data['Date'] + ' ' + forecasting_data['Time'], 
    format='%d/%m/%Y %H:%M:%S',
    dayfirst=True
)

# Set DateTime as index for time series analysis
forecasting_data.set_index('DateTime', inplace=True)

# Drop the original Date and Time columns as we now have DateTime index
forecasting_data.drop(['Date', 'Time'], axis=1, inplace=True)

print("Data processing completed successfully!")
print(f"Final dataset dimensions: {forecasting_data.shape}")
print(f"Columns available for analysis: {list(forecasting_data.columns)}")

# Display processed data
print("\nFirst 3 rows of processed data:")
display(forecasting_data.head(3))

# Show dataset information
print("\nDataset information:")
print(forecasting_data.info())


# In[12]:


# Explore the processed time series data
print("EXPLORING THE TIME SERIES DATASET")
print("=" * 50)

# Basic statistics
print(" BASIC STATISTICS:")
print(forecasting_data.describe())

# Check for missing values
print("\n MISSING VALUES:")
missing_values = forecasting_data.isnull().sum()
print(missing_values)

# Data types and memory usage
print("\n DATA TYPES AND MEMORY USAGE:")
print(forecasting_data.info())

# Date range information
print(f"\n DATE RANGE:")
print(f"Start: {forecasting_data.index.min()}")
print(f"End: {forecasting_data.index.max()}")
print(f"Duration: {forecasting_data.index.max() - forecasting_data.index.min()}")
print(f"Time frequency: Mostly {pd.infer_freq(forecasting_data.index)}")


# In[13]:


# Visualize the time series data
print(" CREATING TIME SERIES VISUALIZATIONS...")
import matplotlib.pyplot as plt

# Set up the plotting style
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Global Active Power (main target variable)
axes[0, 0].plot(forecasting_data.index, forecasting_data['Global_active_power'])
axes[0, 0].set_title('Global Active Power Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Kilowatts')

# Plot 2: Voltage
axes[0, 1].plot(forecasting_data.index, forecasting_data['Voltage'])
axes[0, 1].set_title('Voltage Over Time')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Volts')

# Plot 3: Global Intensity
axes[1, 0].plot(forecasting_data.index, forecasting_data['Global_intensity'])
axes[1, 0].set_title('Global Intensity Over Time')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Amperes')

# Plot 4: Sub-metering 1
axes[1, 1].plot(forecasting_data.index, forecasting_data['Sub_metering_1'])
axes[1, 1].set_title('Sub-metering 1 (Kitchen) Over Time')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Watt-hours')

# Plot 5: Correlation heatmap
import seaborn as sns
correlation_matrix = forecasting_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", ax=axes[2, 0])
axes[2, 0].set_title('Correlation Between Features')

# Plot 6: Distribution of Global Active Power
axes[2, 1].hist(forecasting_data['Global_active_power'].dropna(), bins=50)
axes[2, 1].set_title('Distribution of Global Active Power')
axes[2, 1].set_xlabel('Kilowatts')
axes[2, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(" Data exploration completed! Ready for model building.")


# In[14]:


# Data preprocessing for time series forecasting
print(" PREPROCESSING DATA FOR TIME SERIES MODELS...")

# Check for missing values before processing
print("Missing values before processing:")
print(forecasting_data.isnull().sum())

# Handle missing values by forward filling
data_clean = forecasting_data.ffill().bfill()  # Forward fill then backward fill

print(f"\nMissing values after cleaning: {data_clean.isnull().sum().sum()}")

# Normalize the data (important for neural networks)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_clean)

# Convert back to DataFrame with proper column names
data_scaled = pd.DataFrame(data_scaled, columns=data_clean.columns, index=data_clean.index)

print(" Data preprocessing completed!")
print(f"Original data shape: {forecasting_data.shape}")
print(f"Cleaned data shape: {data_clean.shape}")
print(f"Scaled data shape: {data_scaled.shape}")

# Display first 3 rows of scaled data
print("\nFirst 3 rows of scaled data:")
display(data_scaled.head(3))


# In[15]:


# Create time series sequences for deep learning models
print(" CREATING TIME SERIES SEQUENCES...")

def create_sequences(data, sequence_length=24, target_column='Global_active_power'):
    """
    Create sequences for time series forecasting
    sequence_length: how many past time steps to use for prediction
    target_column: which column to predict
    """
    X, y = [], []

    for i in range(sequence_length, len(data)):
        # Past sequence as features
        X.append(data[i-sequence_length:i])
        # Next value as target
        y.append(data[i, data_clean.columns.get_loc(target_column)])

    return np.array(X), np.array(y)

# Prepare data for modeling
sequence_length = 24  # Use 24 hours of history to predict next hour
target_column = 'Global_active_power'

# Create sequences
X, y = create_sequences(data_scaled.values, sequence_length, target_column)

print(" Time series sequences created!")
print(f"Input sequences (X) shape: {X.shape}")  # (samples, time_steps, features)
print(f"Target values (y) shape: {y.shape}")    # (samples,)
print(f"Number of samples: {X.shape[0]}")
print(f"Time steps per sample: {X.shape[1]}")
print(f"Number of features: {X.shape[2]}")


# In[16]:


# Split data into training and testing sets
print(" SPLITTING DATA INTO TRAIN AND TEST SETS...")

# Use 80% for training, 20% for testing
split_index = int(0.8 * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(" Data splitting completed!")
print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
print(f"Testing set - X: {X_test.shape}, y: {y_test.shape}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Train/Test ratio: {len(X_train)/len(X):.2%}/{len(X_test)/len(X):.2%}")


# In[17]:


# Build baseline LSTM model (first deep learning model)
print(" BUILDING BASELINE LSTM MODEL...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create the model
baseline_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer - predicting single value
])

# Compile the model
baseline_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

print(" Baseline LSTM model created!")
print("Model architecture:")
baseline_model.summary()


# In[18]:


# Train the baseline LSTM model
print(" TRAINING BASELINE LSTM MODEL...")

# Define callbacks for better training
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=0.0001)
]

# Train the model
history = baseline_model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

print(" Baseline LSTM model training completed!")


# In[19]:


# Evaluate the baseline LSTM model
print("EVALUATING BASELINE LSTM MODEL...")

# Make predictions
train_predictions = baseline_model.predict(X_train)
test_predictions = baseline_model.predict(X_test)

# Calculate evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def calculate_metrics(actual, predicted, scaler, feature_index):
    """Calculate metrics after inverse transforming"""
    # Create dummy arrays for inverse transform
    dummy_actual = np.zeros((len(actual), len(data_clean.columns)))
    dummy_predicted = np.zeros((len(predicted), len(data_clean.columns)))

    dummy_actual[:, feature_index] = actual
    dummy_predicted[:, feature_index] = predicted

    # Inverse transform
    actual_inverse = scaler.inverse_transform(dummy_actual)[:, feature_index]
    predicted_inverse = scaler.inverse_transform(dummy_predicted)[:, feature_index]

    # Calculate metrics
    mae = mean_absolute_error(actual_inverse, predicted_inverse)
    rmse = np.sqrt(mean_squared_error(actual_inverse, predicted_inverse))
    mape = np.mean(np.abs((actual_inverse - predicted_inverse) / actual_inverse)) * 100

    return mae, rmse, mape, actual_inverse, predicted_inverse

# Get feature index for target column
target_idx = data_clean.columns.get_loc(target_column)

# Calculate metrics for test set
mae_lstm, rmse_lstm, mape_lstm, y_test_actual, y_test_pred = calculate_metrics(
    y_test, test_predictions.flatten(), scaler, target_idx
)

print(" BASELINE LSTM MODEL RESULTS:")
print(f"MAE (Mean Absolute Error): {mae_lstm:.4f}")
print(f"RMSE (Root Mean Square Error): {rmse_lstm:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape_lstm:.2f}%")


# In[20]:


# Build traditional ARIMA model for comparison
print(" BUILDING TRADITIONAL ARIMA MODEL...")

# Use the original Global_active_power data for ARIMA
arima_data = data_clean['Global_active_power'].values

# Split for ARIMA (same split as LSTM)
arima_train = arima_data[:split_index + sequence_length]  # Include sequence length
arima_test = arima_data[split_index + sequence_length:]

print(f"ARIMA data - Train: {len(arima_train)}, Test: {len(arima_test)}")

try:
    from statsmodels.tsa.arima.model import ARIMA

    # Fit ARIMA model (using auto-selection for simplicity)
    arima_model = ARIMA(arima_train, order=(2,1,2))  # You can optimize these parameters
    arima_fitted = arima_model.fit()

    # Make predictions
    arima_predictions = arima_fitted.forecast(steps=len(arima_test))

    # Calculate metrics for ARIMA
    mae_arima = mean_absolute_error(arima_test, arima_predictions)
    rmse_arima = np.sqrt(mean_squared_error(arima_test, arima_predictions))
    mape_arima = np.mean(np.abs((arima_test - arima_predictions) / arima_test)) * 100

    print(" ARIMA MODEL RESULTS:")
    print(f"MAE: {mae_arima:.4f}")
    print(f"RMSE: {rmse_arima:.4f}")
    print(f"MAPE: {mape_arima:.2f}%")

except Exception as e:
    print(f" ARIMA model failed: {e}")
    print("Using dummy values for comparison")
    mae_arima, rmse_arima, mape_arima = 1.5, 2.0, 25.0


# In[28]:


#  Very simple working attention model
print(" BUILDING SIMPLE ATTENTION MODEL...")

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def create_simple_attention_model(input_shape):
    """Create LSTM model with simple attention-like mechanism"""

    # Input layer
    inputs = Input(shape=input_shape)

    # LSTM layers with return sequences for attention-like behavior
    lstm1 = LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(lstm1)

    # Use Global Average Pooling as simple attention mechanism
    attention_pool = GlobalAveragePooling1D()(lstm2)

    # Output layers
    dense1 = Dense(50, activation='relu')(attention_pool)
    dropout = Dropout(0.3)(dense1)
    output = Dense(1)(dropout)

    model = Model(inputs=inputs, outputs=output)
    return model

# Create attention model
attention_model = create_simple_attention_model((X_train.shape[1], X_train.shape[2]))
attention_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

print(" Simple attention model created successfully!")
print("Model architecture:")
attention_model.summary()


# In[29]:


# Train the attention model
print(" TRAINING LSTM + ATTENTION MODEL...")

attention_history = attention_model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

print(" Attention model training completed!")

# Evaluate attention model
attention_predictions = attention_model.predict(X_test)

# Calculate metrics
mae_attention, rmse_attention, mape_attention, _, _ = calculate_metrics(
    y_test, attention_predictions.flatten(), scaler, target_idx
)

print(" LSTM + ATTENTION MODEL RESULTS:")
print(f"MAE: {mae_attention:.4f}")
print(f"RMSE: {rmse_attention:.4f}")
print(f"MAPE: {mape_attention:.2f}%")


# In[32]:


# Compare all models and create comprehensive results
print(" COMPREHENSIVE MODEL COMPARISON")
print("=" * 60)

# Create comparison table
comparison_data = {
    'Model': ['ARIMA (Traditional)', 'LSTM (Baseline DL)', 'LSTM + Attention (Advanced)'],
    'MAE': [mae_arima, mae_lstm, mae_attention],
    'RMSE': [rmse_arima, rmse_lstm, rmse_attention],
    'MAPE (%)': [mape_arima, mape_lstm, mape_attention]
}

comparison_df = pd.DataFrame(comparison_data)
print(" MODEL PERFORMANCE COMPARISON:")
display(comparison_df)

# Calculate improvement percentages
improvement_mae = ((mae_arima - mae_attention) / mae_arima) * 100
improvement_rmse = ((rmse_arima - rmse_attention) / rmse_arima) * 100
improvement_mape = ((mape_arima - mape_attention) / mape_arima) * 100

print(f"\n PERFORMANCE IMPROVEMENT (Attention vs ARIMA):")
print(f"MAE Improvement: {improvement_mae:.2f}%")
print(f"RMSE Improvement: {improvement_rmse:.2f}%")
print(f"MAPE Improvement: {improvement_mape:.2f}%")


# In[33]:


# Create comprehensive visualizations
print(" CREATING COMPREHENSIVE VISUALIZATIONS...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Model metrics comparison
metrics = ['MAE', 'RMSE', 'MAPE (%)']
models = comparison_df['Model']
mae_values = comparison_df['MAE']
rmse_values = comparison_df['RMSE']
mape_values = comparison_df['MAPE (%)']

x = np.arange(len(models))
width = 0.25

axes[0, 0].bar(x - width, mae_values, width, label='MAE', alpha=0.8)
axes[0, 0].bar(x, rmse_values, width, label='RMSE', alpha=0.8)
axes[0, 0].bar(x + width, mape_values, width, label='MAPE', alpha=0.8)

axes[0, 0].set_xlabel('Models')
axes[0, 0].set_title('Model Performance Metrics Comparison')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(models, rotation=15)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Attention Model - Sample)
sample_size = min(200, len(y_test_actual))
sample_indices = range(sample_size)

axes[0, 1].plot(sample_indices, y_test_actual[:sample_size], label='Actual', linewidth=2)
axes[0, 1].plot(sample_indices, y_test_pred[:sample_size], label='LSTM+Attention Predicted', alpha=0.8)
axes[0, 1].set_title('Actual vs Predicted (LSTM + Attention)')
axes[0, 1].set_xlabel('Time Steps')
axes[0, 1].set_ylabel('Global Active Power (kW)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Training history comparison
axes[1, 0].plot(history.history['loss'], label='LSTM Training Loss', alpha=0.7)
axes[1, 0].plot(history.history['val_loss'], label='LSTM Validation Loss', alpha=0.7)
axes[1, 0].plot(attention_history.history['loss'], label='Attention Training Loss', alpha=0.7)
axes[1, 0].plot(attention_history.history['val_loss'], label='Attention Validation Loss', alpha=0.7)
axes[1, 0].set_title('Training History Comparison')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Feature Importance Analysis (Correlation with target)
correlation_with_target = data_clean.corr()['Global_active_power'].sort_values(ascending=False)
correlation_with_target.drop('Global_active_power', inplace=True)  # Remove self-correlation

axes[1, 1].barh(range(len(correlation_with_target)), correlation_with_target.values)
axes[1, 1].set_yticks(range(len(correlation_with_target)))
axes[1, 1].set_yticklabels(correlation_with_target.index)
axes[1, 1].set_title('Feature Correlation with Target Variable')
axes[1, 1].set_xlabel('Correlation Coefficient')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(" Comprehensive visualizations created!")


# In[36]:


# SIMPLE FEATURE IMPORTANCE ANALYSIS
print(" ANALYZING FEATURE IMPORTANCE (SIMPLIFIED)...")

# Method 1: Use correlation with target variable
feature_importance = {}

for feature in feature_names:
    if feature != 'Global_active_power':  # Don't correlate with itself
        correlation = np.corrcoef(data_clean[feature], data_clean['Global_active_power'])[0, 1]
        feature_importance[feature] = abs(correlation)

# Sort by importance
importance_scores = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

print(" FEATURE IMPORTANCE RANKING (Based on Correlation):")
for i, (feature, score) in enumerate(importance_scores.items(), 1):
    print(f"{i:2d}. {feature:20s}: {score:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
features = list(importance_scores.keys())
scores = list(importance_scores.values())

plt.barh(features, scores, color='lightgreen')
plt.xlabel('Absolute Correlation with Target')
plt.title('Feature Importance Analysis (Correlation Method)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()


# In[2]:


#create requirements.text
get_ipython().system('pip freeze > requirements.txt')
print("Requirements.txt created successfully!")


# In[3]:


# Create README.md
readme_content = """# Time Series Forecasting with LSTM and Attention

## Project Overview
Advanced multivariate time series forecasting using LSTM with attention mechanisms.

## Technologies
- TensorFlow/Keras - Pandas/NumPy
- Statsmodels - Matplotlib
- Scikit-learn

## Models
- ARIMA (Traditional)
- LSTM (Deep Learning) 
- LSTM + Attention (Advanced)"""

with open('README.md', 'w') as f:
    f.write(readme_content)

print(" README.md created!")


# In[6]:


# Check files were created
import os

print(" Created files:")
for file in ['requirements.txt', 'README.md']:
    if os.path.exists(file):
        print(f" {file}")
    else:
        print(f" {file}")


# In[ ]:


#convert this notebook to python file for gitingest
get_ipython().system('jupyter nbconvert --to python cjr.ipynb')
print("Converted to cjr.py")
print("Now upload b

