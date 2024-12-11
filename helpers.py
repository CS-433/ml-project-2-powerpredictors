import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MultiTimeSeriesDataset(Dataset):
    def __init__(self, dataset, seq_len=72): # 72 = 3*24 = hours in three days, since otherwise the RNN might have problems ..
        """
        Args:
            datasets (list of numpy.ndarray): List of time series datasets, 
                each of shape (n_hours, n_features).
            seq_len (int): Length of the input sequence, which the LSTM will be able to see
        """
        self.data = []
        assert dataset.shape[0] > seq_len
        for i in range(dataset.shape[0] - seq_len):
            # Create input-output pairs for each dataset
            x = dataset[i:i + seq_len]
            y = dataset[i + seq_len][4]
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(0)

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

    return timestamps

def shift_dataframe_column(df, key, shift):
    """Modify the dataframe in place. The values of the specified 
    key are shifted down for shift > 0 and up for shift <0
    Args:
        df: (pd.DataFrame); The dataframe to be modified
        key: (String); The key of the column
        shift: (int); Shift amount and direction
    """
    
    df[key] = df[key].shift(shift)
    print(df[shift-2:shift+2])
    df = df.dropna(subset=[key]).reset_index(drop=True)
    data = torch.from_numpy(df.to_numpy()) # Convert to torch tensor
    print('_'*80)
    print(f'In this dataset, the power consumption is now shifted in the future by {shift} days,\nto let the network see the values that are predicted by the forecast')
    print('_'*80)
    return data




