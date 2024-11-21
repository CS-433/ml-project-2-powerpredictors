import pandas as pd
import numpy as np

def relnan(df, key):
    return df[key].isna().sum() / len(df[key])

def LoadData(path):
    """Loading the data and adding seperate columns for year month and day
    Args:
        path: string, path to the csv file, assumed format columns:[something_containing_date, temperature, demand]
    Returns:
        data: dataframe with keys [temperature, demans, year, month, day]
    """

    # Loading data
    data = pd.read_csv(r'C:\Users\Max Tost\Desktop\Notebooks\PowerPrediction\ml-project-2-powerpredictors\data\epfl_campus_demand.csv')

    # Get key of column which contains the date
    key_date = data.keys()[0]

    # Make sure they are in datetime format
    data[key_date] = pd.to_datetime(data[key_date])

    # Adding the corresponding columns
    data['year'] = data[key_date].dt.year
    data['month'] = data[key_date].dt.month
    data['day'] = data[key_date].dt.day
    data['hour'] = data[key_date].dt.hour

    # Drop the irrelevant column
    data = data.drop(key_date, axis=1)

    # Average over the hour to get rid of the minutes
    data = (
    data.groupby(['year', 'month', 'day', 'hour'])
    .mean(numeric_only=True)  # Calculate the mean for each group
    .reset_index()  # Reset the index to turn the grouped columns back into regular columns
    )

    return data