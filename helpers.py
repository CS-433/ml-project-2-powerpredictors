import pandas as pd
import datetime 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from datetime import datetime
import time
import torch


def prepare_features(df, target_col, forecast_cols, window_length=168, forecast_horizon=24, include_forecast=True):
    """
    Prepares the feature and target tensors for TCN with integrated forecasted features,
    filling missing values instead of dropping them.

    Parameters:
        df (pd.DataFrame): Input DataFrame with historical data.
        target_col (str): Column name for power consumption (target variable).
        forecast_cols (list): List of column names for variables for which the forecast is known.
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
        for col in forecast_cols:
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

def picp(y_true, y_lower, y_upper):
    """
    Computes Prediction Interval Coverage Probability (PICP).
    
    Parameters:
        y_true (array-like): True values of the target variable.
        y_lower (array-like): Lower bounds of the prediction intervals.
        y_upper (array-like): Upper bounds of the prediction intervals.
    
    Returns:
        float: Prediction Interval Coverage Probability.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)
    
    # Proportion of true values within the bounds
    coverage = (y_true >= y_lower) & (y_true <= y_upper)  # Boolean array
    picp = np.mean(coverage)  # Average of the boolean array
    
    return picp


def pinaw(y_true, y_lower, y_upper):
    """
    Computes Prediction Interval Normalized Average Width (PINAW).
    
    Parameters:
        y_true (array-like): True values of the target variable.
        y_lower (array-like): Lower bounds of the prediction intervals.
        y_upper (array-like): Upper bounds of the prediction intervals.
    
    Returns:
        float: Prediction Interval Normalized Average Width.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)
    
    # Average width of intervals, normalized by the range of y_true
    interval_widths = y_upper - y_lower
    pinaw = np.mean(interval_widths) / (np.max(y_true) - np.min(y_true))
    
    return pinaw


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

    

def align_predictions_to_timestamps(predictions, timestamps):
    """
    Aligns predictions to their corresponding timestamps for time-based evaluation.

    Parameters:
        predictions (np.ndarray): Predicted values of shape (num_samples, forecast_horizon).
        timestamps (pd.DatetimeIndex): Timestamps for the predictions.

    Returns:
        pd.DataFrame: A DataFrame where predictions are aligned with timestamps.
    """
    num_predictions, forecast_horizon = predictions.shape  # Get number of samples and forecast horizon

    # Initialize an array to hold the aligned predictions with NaN padding
    aligned_predictions = np.full((num_predictions, num_predictions + forecast_horizon - 1), np.nan)

    # Align each prediction to its corresponding position in the timeline
    for sample_idx in range(num_predictions):
        aligned_predictions[sample_idx, sample_idx:sample_idx + forecast_horizon] = predictions[sample_idx]

    # Convert to DataFrame with timestamps as column names
    return pd.DataFrame(aligned_predictions, columns=timestamps)
    

def extract_hourly_data(predictions, prediction_timestamps):
    """
    Extract hourly data from aligned predictions.

    Parameters:
        predictions (pd.DataFrame): Prediction values.
        prediction_timestamps (pd.DatetimeIndex): Timestamps for the predictions.

    Returns:
        pd.DataFrame: Filtered predictions for each hour.
    """
    # Align predictions with timestamps
    aligned_predictions = align_predictions_to_timestamps(predictions, prediction_timestamps)
    aligned_predictions.columns = pd.to_datetime(aligned_predictions.columns)  # Ensure columns are datetime

    # Filter and organize predictions by hour
    hourly_filtered_predictions = []
    for hour in range(24):  
        hourly_data = (aligned_predictions.loc[:, aligned_predictions.columns.hour == hour]).values.flatten()
        hourly_filtered_predictions.append(hourly_data[~np.isnan(hourly_data)])  # Remove NaN values

    # Return as a DataFrame with hours as the index
    return pd.DataFrame(np.vstack(hourly_filtered_predictions), index=range(24))
