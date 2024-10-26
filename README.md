
# Solar Power Forecasting with Neural Networks and Linear Regression

This project implements a neural network model and a linear regression model to forecast solar power output using historical data from Texas and New York. The goal is to demonstrate the effectiveness of deep learning in capturing complex patterns for improved forecasting accuracy, supporting stable and reliable renewable energy grids.

## Table of Contents
- [Setup](#setup)
- [Code Overview](#code-overview)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Model Development and Tuning](#model-development-and-tuning)
- [Model Evaluation and Visualization](#model-evaluation-and-visualization)
- [Results](#results)
- [References](#references)

## Setup

If running in Google Colab, first install the required libraries:
```python
!pip install tensorflow scikit-learn keras-tuner
```

### Required Libraries
- `TensorFlow`: For building and training the neural network model.
- `Scikit-learn`: For data preprocessing, linear regression, and evaluation metrics.
- `Keras Tuner`: For hyperparameter tuning of the neural network.
- `Matplotlib`: For generating visualizations.

Ensure you have the required libraries installed:
```bash
pip install tensorflow scikit-learn keras-tuner matplotlib pandas
```

## Code Overview

This code is structured into three main components:
1. **Data Loading and Preprocessing**: Loads CSV files containing historical power data, merges them based on time, and standardizes features.
2. **Model Development and Tuning**: Builds, tunes, and trains a neural network model. Linear regression is used as a benchmark model for comparison.
3. **Model Evaluation and Visualization**: Evaluates model performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics. Generates several plots to visualize model performance and predictions.

## Data Loading and Preprocessing

### Loading Data
The code reads CSV files containing power output data from Texas and New York. The `load_limited_files` function loads a limited number of CSV files from each location to optimize memory usage.

### Preprocessing
The data is preprocessed to handle missing values with forward-fill and then standardized to a uniform scale using `StandardScaler`. Preprocessed data is then split into training and testing sets for model evaluation.

## Model Development and Tuning

### Neural Network Model
The neural network model is built using Keras and optimized with Keras Tuner:
- **Architecture**: Consists of dense layers with batch normalization and dropout layers to prevent overfitting.
- **Hyperparameter Tuning**: Hyperparameters like the number of units, learning rate, and dropout rate are tuned using random search to minimize validation loss.

### Linear Regression Model
A simple linear regression model is trained as a benchmark for comparing the neural network's effectiveness.

## Model Evaluation and Visualization

### Evaluation Metrics
- **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** are used to evaluate the accuracy of the predictions for both models.

### Visualizations
Several plots are generated to provide insights into model performance:
1. **Neural Network Training and Validation Loss**  
   Shows the training and validation loss over epochs to evaluate the model's learning and generalization ability.

2. **Actual vs. Predicted Power Output**  
   Compares the actual power output with predictions from the neural network and linear regression models.

3. **Error Distribution**  
   Histogram showing the distribution of prediction errors for both models. A narrower distribution suggests more consistent predictions.

4. **Feature Importance (Linear Regression)**  
   Shows the importance of each feature in the linear regression model, giving insight into which variables contribute most to the predictions.

5. **Validation Loss Across Hyperparameter Trials**  
   Shows the validation loss for each trial during hyperparameter tuning, highlighting the tuning's effectiveness in optimizing the neural network.

### Example Usage
The main steps for running the code are outlined in the code cells. Make sure you have the correct folder paths for the Texas and New York data:
```python
tx_folder = 'drive/MyDrive/tx-east-pv-2006'
ny_folder = 'drive/MyDrive/ny-pv-2006'
```

Run each section in sequence, from data loading to model evaluation and visualization.

## Results

The neural network model outperformed linear regression, demonstrating better accuracy and a narrower error distribution. This supports the feasibility of using deep learning for solar power forecasting.

The neural network predictions closely followed the actual power output trends, as shown in the visualizations, which validate its robustness and reliability in real-world forecasting scenarios.

## References
- Kim, H.-S., Park, S., Park, H.-J., Son, H.-G., & Kim, S. (2023). *Solar Radiation Forecasting Based on the Hybrid CNN-CatBoost Model.* IEEE Access.
- Fungtammasan, G., & Koprinska, I. (2023). *Convolutional and LSTM Neural Networks for Solar Power Forecasting.* 2023 International Joint Conference on Neural Networks (IJCNN).

---

This `README.md` provides instructions and details for understanding and running the code. Make sure to adapt folder paths and configurations as needed.
