import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
from scipy import stats

def plot_data_distribution(df, columns, colors, suptitle, sharex_bool, is_violin):
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(8, 5), sharex=sharex_bool)
    if is_violin:
        for i, feature in enumerate(columns):
            sns.violinplot(
                data=df,
                x=feature,
                ax=axes[i],
                color=colors[i],  # Assign unique color from the palette
                scale="width"  # Keep violin widths normalized
            )
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel(feature)
    else:
        for i, feature in enumerate(columns):
            sns.boxplot(
                data=df,
                x=feature,
                ax=axes[i],
                color=colors[i]  # Assign unique color from the palette
            )        
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel(feature)

    plt.suptitle(suptitle)
    
    plt.tight_layout()
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

def plot_residuals_per_hour(hourly_residuals, hourly_predictions):
    """
    Plots 24 rows x 3 columns of residual analysis, one row per hour.
    Shared X-axis for all subplots.

    Parameters:
        hourly_residuals (pd.DataFrame): DataFrame where each row contains residuals for an hour.
        hourly_predictions (pd.DataFrame): DataFrame where each row contains predictions for an hour.
    """
    # Create a 24x3 grid of subplots
    fig, axes = plt.subplots(nrows=24, ncols=3, figsize=(10, 60), sharex=True)

    for hour in range(24):
        residuals = hourly_residuals.iloc[hour].dropna().values
        predictions = hourly_predictions.iloc[hour].dropna().values

        # Residuals vs Predicted Values
        axes[hour, 0].scatter(predictions, residuals, alpha=0.5)
        axes[hour, 0].axhline(0, color='red', linestyle='--')
        axes[hour, 0].set_title(f"Hour {hour+1}: Residuals vs Predicted")
        axes[hour, 0].set_ylabel("Residuals")
        if hour == 23:  # Add X-axis label only to the last row
            axes[hour, 0].set_xlabel("Predicted Values")

        # QQ Plot
        stats.probplot(residuals.flatten(), dist="norm", plot=axes[hour, 1])
        axes[hour, 1].set_title(f"Hour {hour+1}: QQ Plot")

        # Histogram of Residuals
        axes[hour, 2].hist(residuals.flatten(), bins=30, edgecolor='k', alpha=0.7)
        axes[hour, 2].set_title(f"Hour {hour+1}: Histogram")
        if hour == 23:  # Add X-axis label only to the last row
            axes[hour, 2].set_xlabel("Residuals")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_hourly_residuals_distribution(data, title, savename):
    """
    Plots 24 vertical boxplots, one for each hour, from the input DataFrame.

    Parameters:
        data (pd.DataFrame): A DataFrame where each row represents data for an hour.
        title (str): The title of the plot.
        savename (str): File name to save the plot.
    """
    if data.shape[0] != 24:
        raise ValueError("Input DataFrame must have 24 rows, one for each hour.")

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.boxplot(data.values.T, vert=True, patch_artist=True, showfliers=True)
    
    # Customize the plot
    plt.title(title)
    plt.xlabel("Hour of the Day [h]")
    plt.ylabel("Residuals")
    plt.xticks(ticks=np.arange(1, 25), labels=data.index if data.index is not None else np.arange(1, 25))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
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