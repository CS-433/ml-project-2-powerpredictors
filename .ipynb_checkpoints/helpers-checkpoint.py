import pandas as pd
import datetime 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from datetime import datetime
import time


def day_ahead_forecast(forecast_np, prediction_timestamps):
    """
    Generate direct day-ahead forecasts.
    Predicts at time k the next k+24 hours and aligns predictions with test timestamps.
    """
    num_samples, forecast_horizon = forecast_np.shape  

    forecast = []
    # Loop through the test set in steps of 24 (non-overlapping)
    for i in range(0, num_samples, forecast_horizon):
        forecast.append(forecast_np[i])
    
    # Convert predictions and timestamps to a Pandas Series
    forecast_series = pd.Series(
        np.concatenate(forecast),  # Flatten the list of forecasts
        index=prediction_timestamps  # Align with timestamps
    )
    return forecast_series

def rolling_forecast(fitted_model, test, exog_test, forecast_horizon=24):
    """
    Generate rolling day-ahead forecasts and align them with timestamps.
    Aligns predictions and computes the mean for overlapping forecasts.
    """
    # Dictionary to store all predictions for each time step
    aligned_predictions = {t: [] for t in test.index}

    # Generate rolling forecasts
    for i in range(0, len(test) - forecast_horizon + 1):  # Overlapping windows
        # Forecast the next 24 hours
        forecast = fitted_model.forecast(
            steps=forecast_horizon,
            exog=exog_test.iloc[i:i + forecast_horizon]
        )
        forecast_times = test.index[i:i + forecast_horizon]  # Get the corresponding times
        
        # Align each forecast with its timestamp
        for t, pred in zip(forecast_times, forecast):
            aligned_predictions[t].append(pred)

    # Compute the mean of overlapping forecasts for each time step
    aggregated_predictions = pd.Series({
        t: np.mean(preds) for t, preds in aligned_predictions.items() if preds
    })

    return aggregated_predictions

def prepare_tcn_features(df, target_col, exog_cols, window_length=168, forecast_horizon=24, include_forecast=True):
    """
    Prepares the feature and target tensors for TCN with integrated forecasted features,
    filling missing values instead of dropping them.

    Parameters:
        df (pd.DataFrame): Input DataFrame with historical data.
        target_col (str): Column name for power consumption (target variable).
        exog_cols (list): List of column names for variables for which the forecast is known.
        window_length (int): Length of the historical temporal window (e.g., 168 for 7 days).
        forecast_horizon (int): Forecast horizon (e.g., 24 for next 24 hours).

    Returns:
        np.ndarray: Feature tensor of shape (num_samples, window_length, num_features).
        np.ndarray: Target tensor of shape (num_samples, forecast_horizon).
        pd.DatetimeIndex: Timestamps corresponding to each sample.
    """
    df = df.copy()

    if include_forecast:
        # Add forecasted weather features
        for col in exog_cols:
            for h in range(1, forecast_horizon + 1):
                df[f'{col}_forecast_{h}h'] = df[col].shift(-h)
    
        # Fill missing values (forward fill and backward fill as fallback)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    # Prepare features and targets
    X, y = [], []
    timestamps = []

    for i in range(len(df) - window_length - forecast_horizon + 1):
        # Extract historical and forecasted features
        features = df.iloc[i:i + window_length].values  # (window_length, num_features)
        X.append(features)

        # Extract target (next 24 hours of power consumption)
        y.append(df.iloc[i + window_length:i + window_length + forecast_horizon][target_col].values)

        # Timestamps for the target period
        timestamps.append(df.index[i + window_length])

    return np.array(X), np.array(y), pd.DatetimeIndex(timestamps)

def compute_picp_pinaw(y_true, y_lower, y_upper):
    """
    Computes PICP (Prediction Interval Coverage Probability) and 
    PINAW (Prediction Interval Normalized Average Width).
    
    Parameters:
        y_true (array-like): True values of the target variable.
        y_lower (array-like): Lower bounds of the prediction intervals.
        y_upper (array-like): Upper bounds of the prediction intervals.
    
    Returns:
        picp (float): Prediction Interval Coverage Probability.
        pinaw (float): Prediction Interval Normalized Average Width.
    """
    # Convert inputs to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)
    
    # PICP: Proportion of true values within the bounds
    coverage = (y_true >= y_lower) & (y_true <= y_upper)  # Boolean array
    picp = np.mean(coverage)  # Average of the boolean array
    
    # PINAW: Average width of intervals, normalized by the range of y_true
    interval_widths = y_upper - y_lower
    pinaw = np.mean(interval_widths) / (np.max(y_true) - np.min(y_true))
    
    return picp, pinaw