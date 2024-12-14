import pandas as pd
import datetime 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from datetime import datetime
import time
import torch


def prepare_features(df, target_col, exog_cols, window_length=168, forecast_horizon=24, include_forecast=True):
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

    
def extract_residuals(y_true, y_pred, prediction_timestamps, hours):

    y_true_filtered = [] 
    y_pred_filtered = []
    residuals_filtered = []
    
    for hour in hours:
        y_true_filtered_temp = (y_true[y_true.index.hour == hour]).to_numpy()
        y_pred_filtered_temp = (y_pred[y_pred.index.hour == hour]).to_numpy()
        y_true_filtered.append(y_true_filtered_temp)
        y_pred_filtered.append(y_pred_filtered_temp)
        residuals_filtered.append(y_true_filtered_temp-y_pred_filtered_temp)
        
    return pd.DataFrame(np.vstack(y_true_filtered), index = hours), pd.DataFrame(np.vstack(y_pred_filtered), index = hours), pd.DataFrame(np.vstack(residuals_filtered), index = hours)

def time_align_predictions(y_pred, prediction_timestamps):
    
    num_samples, forecast_horizon = y_pred.shape  

    aligned_pred = np.full((num_samples, num_samples+forecast_horizon-1), np.nan)
    for i in range(num_samples):  
        aligned_pred[i, i:i + forecast_horizon] = y_pred[i] 
    
    return pd.DataFrame(aligned_pred, columns = prediction_timestamps) 
    
def extract_all_residuals(y_true_all, y_pred_all, prediction_timestamps, hours):

    y_true_all_aligned = time_align_predictions(y_true_all, prediction_timestamps)
    y_true_all_aligned.columns = pd.to_datetime(y_true_all_aligned.columns)  
    y_pred_all_aligned = time_align_predictions(y_pred_all, prediction_timestamps)
    y_pred_all_aligned.columns = pd.to_datetime(y_pred_all_aligned.columns)  

    y_true_all_filtered = []
    y_pred_all_filtered = []
    residuals_all_filtered = []
    for hour in hours:
        filtered_y_true_all_aligned = (y_true_all_aligned.loc[:, y_true_all_aligned.columns.hour == hour]).values.flatten()
        filtered_y_pred_all_aligned = (y_pred_all_aligned.loc[:, y_pred_all_aligned.columns.hour == hour]).values.flatten()

        y_true_all_filtered.append(filtered_y_true_all_aligned[~np.isnan(filtered_y_true_all_aligned)])
        y_pred_all_filtered.append(filtered_y_pred_all_aligned[~np.isnan(filtered_y_pred_all_aligned)])
        residuals_all_filtered.append(filtered_y_true_all_aligned[~np.isnan(filtered_y_true_all_aligned)]-filtered_y_pred_all_aligned[~np.isnan(filtered_y_pred_all_aligned)])
        
    return pd.DataFrame(np.vstack(y_true_all_filtered), index = hours), pd.DataFrame(np.vstack(y_pred_all_filtered), index = hours), pd.DataFrame(np.vstack(residuals_all_filtered), index = hours)

