# Install necessary libraries (if running in Colab)
!pip install tensorflow scikit-learn keras-tuner

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras_tuner as kt

tx_folder = 'drive/MyDrive/tx-east-pv-2006'
ny_folder = 'drive/MyDrive/ny-pv-2006'

def load_limited_files(folder_path, keyword, usecols=None, file_limit=5):
    """Load a limited number of CSV files to optimize memory usage."""
    dataframes = []
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and keyword in filename:
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, usecols=usecols)
            dataframes.append(df)
            file_count += 1
            if file_count >= file_limit:
                break
    return pd.concat(dataframes, ignore_index=True)

use_columns = ['LocalTime', 'Power(MW)']
tx_actual_data = load_limited_files(tx_folder, 'Actual', usecols=use_columns)
tx_da_data = load_limited_files(tx_folder, 'DA', usecols=use_columns)
tx_ha4_data = load_limited_files(tx_folder, 'HA4', usecols=use_columns)
ny_actual_data = load_limited_files(ny_folder, 'Actual', usecols=use_columns)
ny_da_data = load_limited_files(ny_folder, 'DA', usecols=use_columns)
ny_ha4_data = load_limited_files(ny_folder, 'HA4', usecols=use_columns)

tx_data = pd.merge(pd.merge(tx_actual_data, tx_da_data, on='LocalTime'), tx_ha4_data, on='LocalTime')
ny_data = pd.merge(pd.merge(ny_actual_data, ny_da_data, on='LocalTime'), ny_ha4_data, on='LocalTime')

def preprocess_data(data):
    """Preprocess and scale data, filling in missing values."""
    data = data.fillna(method='ffill')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(['LocalTime'], axis=1))
    return pd.DataFrame(scaled_data, columns=data.columns.drop(['LocalTime']))

tx_data_processed = preprocess_data(tx_data)
ny_data_processed = preprocess_data(ny_data)
target_column = 'Power(MW)'

if target_column in tx_data_processed.columns:
    X_train_tx, X_test_tx, y_train_tx, y_test_tx = train_test_split(
        tx_data_processed.drop([target_column], axis=1),
        tx_data_processed[target_column], test_size=0.2, random_state=42
    )
else:
    raise ValueError(f"Target column '{target_column}' not found in data")

def build_model(hp):
    model = models.Sequential()
    model.add(layers.Dense(units=hp.Int('units_1', min_value=32, max_value=128, step=16), activation='relu', input_shape=(X_train_tx.shape[1],)))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mse')
    return model

tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=5, directory='my_dir', project_name='neural_network_tuning')
tuner.search(X_train_tx, y_train_tx, epochs=20, validation_split=0.2, callbacks=[callbacks.EarlyStopping(patience=3)])

model = tuner.get_best_models(num_models=1)[0]
history = model.fit(X_train_tx, y_train_tx, validation_split=0.2, epochs=20, batch_size=128, callbacks=[callbacks.EarlyStopping(patience=3)])

X_ny = ny_data_processed.drop([target_column], axis=1)
y_ny = ny_data_processed[target_column]
ny_predictions_nn = model.predict(X_ny)

mse_ny_nn = mean_squared_error(y_ny, ny_predictions_nn)
mae_ny_nn = mean_absolute_error(y_ny, ny_predictions_nn)

try:
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train_tx, y_train_tx)
    y_pred_ny_lr = linear_reg_model.predict(X_ny)
    
    mse_ny_lr = mean_squared_error(y_ny, y_pred_ny_lr)
    mae_ny_lr = mean_absolute_error(y_ny, y_pred_ny_lr)
except Exception as e:
    raise RuntimeError(f"Linear Regression Model error: {e}")

mse_comparison = {
    "Neural Network MSE on NY Data": mse_ny_nn,
    "Neural Network MAE on NY Data": mae_ny_nn,
    "Linear Regression MSE on NY Data": mse_ny_lr,
    "Linear Regression MAE on NY Data": mae_ny_lr
}
print(pd.DataFrame(mse_comparison, index=["Metrics"]))

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Neural Network Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(ny_data_processed.index, y_ny, label='Actual Power (NY)', color='blue', linewidth=2)
plt.plot(ny_data_processed.index, ny_predictions_nn, label='Neural Network Predictions (NY)', color='green', linestyle='--')
plt.plot(ny_data_processed.index, y_pred_ny_lr, label='Linear Regression Predictions (NY)', color='red', linestyle='-.')
plt.xlabel('Time')
plt.ylabel('Power (MW)')
plt.legend()
plt.title('Actual vs. Predicted Power in NY')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(14, 6))
plt.hist(y_ny - ny_predictions_nn.flatten(), bins=30, alpha=0.6, label='Neural Network Errors', color='green', edgecolor='black')
plt.hist(y_ny - y_pred_ny_lr, bins=30, alpha=0.6, label='Linear Regression Errors', color='red', edgecolor='black')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Error Distribution for Neural Network and Linear Regression Models')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

feature_importance = pd.Series(linear_reg_model.coef_, index=X_train_tx.columns).sort_values()
plt.figure(figsize=(12, 6))
feature_importance.plot(kind='bar', color='steelblue', edgecolor='black')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Importance (Linear Regression)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

val_losses = [trial.metrics.get_best_value('val_loss') for trial in tuner.oracle.get_best_trials(num_trials=5)]
plt.figure(figsize=(10, 6))
plt.bar(range(len(val_losses)), val_losses, color='skyblue', edgecolor='black')
plt.xlabel('Trial')
plt.ylabel('Best Validation Loss')
plt.title('Validation Loss Across Hyperparameter Trials')
plt.xticks(range(len(val_losses)), [f'Trial {i+1}' for i in range(len(val_losses))])
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
