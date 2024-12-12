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
    print('-'*80)
    return data


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob):
        """
        Initialize the LSTM-based regression model.

        Args:
            input_size (int): Number of input features (e.g., temperature, GHI, etc.).
            hidden_size (int): Number of units in each LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Number of output features (e.g., predicted demand, 1 for regression).
            dropout_prob (float): Dropout probability to apply between LSTM layers and before the fully connected layer.
        """
        super(LSTMModel, self).__init__()

        # LSTM Layer
        # - Processes sequential data and learns temporal dependencies.
        # - Supports multiple layers (num_layers) and applies dropout between layers.
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Fully Connected (Linear) Layer
        # - Maps the LSTM's hidden state output to the desired output size.
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout Layer
        # - Reduces overfitting by randomly zeroing some activations during training.
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Forward pass for the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size).
        """
        # LSTM Layer
        # - Returns the full sequence of hidden states and the final hidden/cell state tuple.
        # - We ignore the hidden/cell state tuple here (h_n, c_n).
        _ , (h_f, _) = self.lstm(x)

        # Dropout Layer
        # - Only uses the hidden state from the last time step for prediction.
        # - Applies dropout to prevent overfitting.
        out = self.dropout(h_f[-1])  # Only using the last hidden state, since they are passed forward between the lstms; Shape: (batch_size, hidden_size)

        # Fully Connected Layer
        # - Maps the LSTM's output to the desired output size (e.g., single regression output).
        out = self.fc(out)  # Shape: (batch_size, output_size)

        return out
    
def train_lstm(model, criterion, optimizer, train_loader, val_loader, num_epochs, scheduler):
    # A single validation step before training
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            val_output = model(x_val)
            val_loss += criterion(val_output, y_val).item()
    val_loss /= len(val_loader)
    print(f"Before training, Validation Loss with random parameters: {val_loss:.4f}")

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()

            # Get output
            output = model(x)

            # Compute loss
            loss = criterion(output, y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Step the scheduler at the end of the epoch
        scheduler.step()

        # Validation (optional)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_output = model(x_val)
                val_loss += criterion(val_output, y_val).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        # Print the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.6f}")

def LoadData(path, look_ahead):
    """
    Loads data and shift the data in the dataframe (from the csv at the indicated path) by the amount of hours indicated in the future

    Args:
        path (String): Path to the data, including the normalizes power consumption values 
        look_ahead: The number of hours that the power_consumption will be shifted in the future (for positive values, recommended, standard = 5) or to the past (doesnt make sense, dont to it)
    Returns:
        train_loader (Torch DataLoader): Dataloader containing the first 80% of the timeseries data
        val_loader (Torch DataLoader): Containing the rest
        data_tensor (Torch tensor): Containing all the data
    """
    data = pd.read_csv(path)
    print(data[:3])
    data_tensor = shift_dataframe_column(data, 'power_consumption', look_ahead)
    print(data_tensor[0])

    # Creating datasets
    train_dataset = MultiTimeSeriesDataset(data_tensor[:int(0.8*len(data_tensor))]) # Using first 80% for training
    val_dataset = MultiTimeSeriesDataset(data_tensor[int(0.8*len(data_tensor)):]) # Using last 20% for evaluation 

    # DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    return train_loader, val_loader, data_tensor



