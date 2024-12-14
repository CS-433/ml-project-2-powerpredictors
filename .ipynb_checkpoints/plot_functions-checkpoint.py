import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
from scipy import stats

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
    
def plot_results_by_month(pred, act, title, savename):

    pred.index = pd.to_datetime(pred.index)
    act.index = pd.to_datetime(act.index)

    # Group data by month
    months = pred.index.month.unique()
    num_months = len(months)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_months, ncols=1, figsize=(20, 4 * num_months), sharex=False)
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
    plt.savefig(savename)
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

def plot_results_with_uncertainty_by_month(mean_predictions, uncertainties, confidence_level, actuals, title, savename):
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
    fig, axes = plt.subplots(nrows=num_months, ncols=1, figsize=(20, 4 * num_months), sharex=False)
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
    plt.savefig(savename)
    plt.show()

def plot_results_with_uncertainty_by_week(mean_predictions, uncertainties, confidence_level, actuals, title, savename):
    """
    Plots weekly results with uncertainty intervals.

    Parameters:
        mean_predictions (pd.Series): Mean predictions with datetime index.
        uncertainties (pd.Series): Uncertainty values with the same index as mean_predictions.
        confidence_level (float): Multiplier for confidence interval.
        actuals (pd.Series): True values with datetime index.
        title (str): Overall plot title.
        savename (str): Path to save the plot.
    """
    # Ensure datetime indices
    mean_predictions.index = pd.to_datetime(mean_predictions.index)
    uncertainties.index = pd.to_datetime(uncertainties.index)
    actuals.index = pd.to_datetime(actuals.index)

    # Group data by week
    weeks = mean_predictions.index.to_series().dt.isocalendar().week.unique()
    num_weeks = len(weeks)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_weeks, ncols=1, figsize=(20, 4 * num_weeks), sharex=False)
    if num_weeks == 1:  # If there's only one week, axes is not a list
        axes = [axes]

    for ax, week in zip(axes, weeks):
        # Filter data for the current week
        mean_week = mean_predictions[mean_predictions.index.to_series().dt.isocalendar().week == week]
        uncertainty_week = uncertainties[uncertainties.index.to_series().dt.isocalendar().week == week]
        actual_week = actuals[actuals.index.to_series().dt.isocalendar().week == week]

        # Plot the data
        ax.plot(actual_week.index, actual_week, label="True", color="red")
        ax.plot(mean_week.index, mean_week, label="Mean Prediction", color="blue")
        ax.fill_between(
            mean_week.index,
            mean_week - confidence_level * uncertainty_week,
            mean_week + confidence_level * uncertainty_week,
            color="blue",
            alpha=0.3,
            label="Confidence Interval"
        )

        # Customize the subplot
        ax.set_title(f"Week {week}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power Consumption")
        ax.legend()
        ax.grid()

    # Add overall title and layout adjustment
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  
    plt.savefig(savename)
    plt.show()

def plot_residuals(residuals, y_pred, title, savename):
    """
    Plot residual analysis including:
    - Residuals vs Predicted Values
    - QQ Plot of Residuals
    - Histogram of Residuals
    
    Parameters:
        residuals (np.ndarray): Residual values (actual - predicted).
        y_pred (np.ndarray): Predicted values.
        hour (int): Hour of the day for context in the plot title.
        savename (str): File name to save the plot.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Residuals vs Predicted Values
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted Values")

    # QQ Plot
    stats.probplot(residuals.flatten(), dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot of Residuals")

    # Histogram of Residuals
    axes[2].hist(residuals.flatten(), bins=30, edgecolor='k', alpha=0.7)
    axes[2].set_xlabel("Residuals")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Histogram of Residuals")

    # Adjust layout
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(savename)
    plt.show()


def plot_train_vs_validation_loss(epochs,train_losses, val_losses, savename):
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')  # Red line with markers
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')  # Blue line with markers
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(savename)
    plt.show()