import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import datetime

def plot_results(pred, act, title):

    pred.index = pd.to_datetime(pred.index)
    act.index = pd.to_datetime(act.index)
    
    plt.figure(figsize=(12, 6))
    plt.plot(act.index, act, label="True")
    plt.plot(pred.index, pred, label="Predicted", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power Consumption")
    plt.grid()
    plt.show()
    
def plot_results_by_month(pred, act, title):

    pred.index = pd.to_datetime(pred.index)
    act.index = pd.to_datetime(act.index)

    # Group data by month
    months = pred.index.month.unique()
    num_months = len(months)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_months, ncols=1, figsize=(50, 4 * num_months), sharex=False)
    if num_months == 1:  # If there's only one month, axes is not a list
        axes = [axes]
    
    for ax, month in zip(axes, months):
        # Filter data for the current month
        pred_month = pred[pred.index.month == month]
        act_month = act[act.index.month == month]
        
        # Plot the data
        ax.plot(act_month.index, act_month, label="True")
        ax.plot(pred_month.index, pred_month, label="Predicted", linestyle="--")
        
        # Customize the subplot
        ax.set_title(f"Month: {month}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power Consumption")
        ax.legend()
        ax.grid()

    # Add overall title and layout adjustment
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  
    plt.show()


def plot_results_with_uncertainty(mean_predictions, uncertainties, confidence_level, actuals, title):

    time_steps = pd.to_datetime(mean_predictions.index)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, actuals, label="True", color="red")
    plt.plot(time_steps, mean_predictions, label="Mean Prediction", color="blue")
    plt.fill_between(
        time_steps,
        mean_predictions - confidence_level * uncertainties,
        mean_predictions + confidence_level * uncertainties,
        color="blue",
        alpha=0.3,
        label="95% Confidence Interval"
    )
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.show()

def plot_results_with_uncertainty_by_month(mean_predictions, uncertainties, confidence_level, actuals, title):
    """
    Plots monthly results with uncertainty intervals.
    
    Parameters:
        mean_predictions (pd.Series): Mean predictions with datetime index.
        uncertainties (pd.Series): Uncertainty values with the same index as mean_predictions.
        confidence_level (float): Multiplier for confidence interval.
        actuals (pd.Series): True values with datetime index.
        title (str): Overall plot title.
    """
    mean_predictions.index = pd.to_datetime(mean_predictions.index)
    uncertainties.index = pd.to_datetime(uncertainties.index)
    actuals.index = pd.to_datetime(actuals.index)

    # Group data by month
    months = mean_predictions.index.month.unique()
    num_months = len(months)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_months, ncols=1, figsize=(15, 5 * num_months), sharex=False)
    if num_months == 1:  # If there's only one month, axes is not a list
        axes = [axes]
    
    for ax, month in zip(axes, months):
        # Filter data for the current month
        mean_month = mean_predictions[mean_predictions.index.month == month]
        uncertainty_month = uncertainties[uncertainties.index.month == month]
        actual_month = actuals[actuals.index.month == month]
        
        # Plot the data
        ax.plot(actual_month.index, actual_month, label="True", color="red")
        ax.plot(mean_month.index, mean_month, label="Mean Prediction", color="blue")
        ax.fill_between(
            mean_month.index,
            mean_month - confidence_level * uncertainty_month,
            mean_month + confidence_level * uncertainty_month,
            color="blue",
            alpha=0.3,
            label="Confidence Interval"
        )
        
        # Customize the subplot
        ax.set_title(f"Month: {month}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power Consumption")
        ax.legend()
        ax.grid()

    # Add overall title and layout adjustment
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  
    plt.show()

def plot_curves_with_reference(x, ref_curve, curves, labels, colors, title="Plot with Reference and Curves"):

    plt.figure(figsize=(10, 6))
    
    # Plot the reference curve (solid line)
    plt.plot(x, ref_curve, color="black", label="Reference Curve", linewidth=2)
    
    # Plot each dashed curve and fill the area between the reference and the curve
    for curve, label, color in zip(curves, labels, colors):
        plt.plot(x, curve, linestyle="--", color=color, label=label, linewidth=1.5)
        plt.fill_between(x, ref_curve, curve, color=color, alpha=0.3)
    
    # Add labels, legend, and title
    plt.xlabel("Time")
    plt.ylabel("Power consumption")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_curves_with_reference_by_month(x, ref_curve, curves, labels, colors, title="Plot with Reference and Curves by Month"):
    
    if len(curves) != len(labels) or len(curves) != len(colors):
        raise ValueError("The number of curves, labels, and colors must match.")

    # Ensure x, ref_curve, and curves are indexed by timestamps
    if not isinstance(x, pd.DatetimeIndex):
        raise TypeError("x must be a pd.DatetimeIndex.")
    
    # Group data by month
    months = x.to_series().groupby(x.month).unique()
    num_months = len(months)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_months, ncols=1, figsize=(50, 5 * num_months), sharex=False)
    if num_months == 1:  # If there's only one month, axes is not a list
        axes = [axes]
    
    for ax, (month, timestamps) in zip(axes, months.items()):
        # Filter data for the current month
        ref_month = ref_curve.loc[timestamps]
        curves_month = [curve.loc[timestamps] for curve in curves]

        # Plot the reference curve
        ax.plot(ref_month.index, ref_month, color="black", label="Reference Curve", linewidth=2)

        # Plot prediction curves and fill the areas
        for curve, label, color in zip(curves_month, labels, colors):
            ax.plot(curve.index, curve, linestyle="--", color=color, label=label, linewidth=1.5)
            ax.fill_between(curve.index, ref_month, curve, color=color, alpha=0.3)

        # Customize subplot
        ax.set_title(f"Month: {pd.to_datetime(ref_month.index[0]).strftime('%B')}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power Consumption")
        ax.legend()
        ax.grid(True)

    # Add overall title and adjust layout
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the title
    plt.show()