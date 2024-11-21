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
    data['year'] = data['Unnamed: 0'].dt.year
    data['month'] = data['Unnamed: 0'].dt.month
    data['day'] = data['Unnamed: 0'].dt.day

    # Drop the irrelevant column
    data.drop(key_date, axis=1)

    return data