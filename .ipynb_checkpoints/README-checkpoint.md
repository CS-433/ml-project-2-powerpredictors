# Power Consumption Forecasting Repository

This repository contains the implementation of models and methods for day-ahead forecast of power consumption using advanced techniques such as Long Short Term Memory networks (LSTMs) and Temporal Convolutional Networks (TCNs). The project also includes data preprocessing, feature engineering, and benchmark model evaluation.

## Project Structure

### **Folders**
- **`data/`**: Contains raw data used for the project.
- **`data_augmented/`**: Contains cleaned and pre-processed data sets with additional engineered features.
- **`results/`**: Stores results and plots generated during the model evaluation.

### **Notebooks**
1. **`data_cleaning.ipynb`**: Handles data cleaning tasks such as missing value imputation, outlier removal, feature augmentation.
2. **`data_preprocessing.ipynb`**: Focuses on preparing the data for use in machine learning models (e.g., train-test splitting, standardization and one-hot encoding, data structuring).
3. **`benchmark.ipynb`**: Implements and evaluates benchmark models, namely SARIMA and Persistence Forecast models.
4. **`LSTM.ipynb`**: Implements and trains an LSTM model for power consumption forecasting.
5. **`TCN.ipynb`**: Implements and trains the TCN model for power consumption forecasting.
6. **`results.ipynb`**: Summarizes and visualizes results from the various models, including comparison metrics and plots.

### **Python Scripts**
- **`helpers.py`**: Utility functions for data processing, feature engineering, and other reusable code snippets.
- **`plot_functions.py`**: Functions for generating visualizations, including residual plots, loss curves, and model performance comparisons.

### **README.md**
Provides an overview of the repository, including its structure and usage.

---

## Project Description
This project aims to forecast power consumption using both heuristic, standard regression and machine learning approaches. It incorporates:
- Extensive data cleaning, augmentation and feature engineering.
- Benchmark models (SARIMA, Persistence Forecast).
- Advanced machine learning models (LSTM, TCN).

The project is designed to predict power consumption for the next 24 hours (day-ahead forecasting), leveraging weather forecast data, time-based features, and historical trends.

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/CS-433/ml-project-2-powerpredictors.git
   cd ml-project-2-powerpredictors
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing pipeline:
   - Open and run `data_cleaning.ipynb` to clean the dataset.
   - Open and run `data_preprocessing.ipynb` to prepare the dataset for model training and evaluation.
     
4. Train and test the benchmark models:
   - Open and run  `benchmark.ipynb` to train and test the SARIMA and Persistence Forecast models.
    
5. Train and test the advance neural network models:
   - Open and run `LSTM.ipynb` to train and test the LSTM model, .
   - Open and run `TCN.ipynb` to train and test the TCN model.

6. Evaluate results:
   - Open and run `results.ipynb` to display the prediction accuracy and uncertainty using quantitative metrics and visualization tools.

---

## Results
The models are evaluated, and the results are presented through:  
- Comparative performance metrics, namely RMSE, MAE, ME, and MAPE for prediction accuracy, and PICP and PINAW for quantifying prediction uncertainty.  
- Goodness-of-fit plots, including train vs validation loss curves.  
- Validation of model assumptions, e.g. residual plots to assess model validity.  
- Time series visualization, namely predicted vs actual values plotted over time.  

---

