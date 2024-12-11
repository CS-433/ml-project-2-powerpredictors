import pandas as pd
import numpy as np

def relnan(df, key):
    """Giving the relative Nan values for a key in the dataframe df"""
    return df[key].isna().sum() / len(df[key])

def LoadTimestamps(path_timestamps):
    """
    Loading the prepared timestamps.csv data to use merge them with the rest of the data.
    """
    timestamps = pd.read_csv(path_timestamps)

    # Get key of column which contains the date
    key_date = timestamps.keys()[0]

    # Make sure they are in datetime format
    key_date = timestamps.keys()[1]
    timestamps[key_date] = pd.to_datetime(timestamps[key_date])

    # Adding the corresponding columns
    timestamps['year'] = timestamps[key_date].dt.year
    timestamps['month'] = timestamps[key_date].dt.month
    timestamps['day'] = timestamps[key_date].dt.day
    timestamps['hour'] = timestamps[key_date].dt.hour

    # Drop the irrelevant column
    timestamps = timestamps.drop(key_date, axis=1)

    # Key for unnamed column
    key_unnamed = timestamps.keys()[0]
    timestamps = timestamps.drop(key_unnamed, axis=1) # Dropt it like its hot!
     
    return timestamps



def LoadData(path):
    """Loading the data and adding seperate columns for year month and day
    Args:
        path: string, path to the csv file, assumed format columns:[something_containing_date, temperature, demand]
    Returns:
        data: dataframe with keys [temperature, demans, year, month, day]
    """

    # Loading data
    data = pd.read_csv(path)

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

def return_nan_sections(data, key):
    """Returns the first and last index of the data frame where the values are nan
    Args:
        data: dataframe containing a column where there are nan values inside
        key: string, the key of the column where there are nans
    Returns:
        nan_sections: nparray, contains first and last index where value is nan
    """
    nan_sections = []
    nan_indices = data[key][data[key].isna()].index  # Get indices of NaN values
    if len(nan_indices) > 0:  # Ensure there are NaN values to process
        first_nan = nan_indices[0]  # Start of the first NaN section
        temp = nan_indices[0]  # Temp to track contiguous indices

        for n in nan_indices[1:]:  # Loop through remaining indices
            if n - 1 != temp:  # Check if current index is not contiguous with the previous
                nan_sections.append([first_nan, temp+1])  # Append the current section
                first_nan = n  # Start a new section
            temp = n  # Update temp to current index

        # Add the last section
        nan_sections.append([first_nan, temp])

    return np.array(nan_sections)