import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import joblib
import logging

# Configure logging
logging.basicConfig(
    filename="store_sales_prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class StoreSalesPrediction:
    def __init__(self, train_path, test_path, store_path):
        """
        Initialize the class with file paths.
        Args:
            train_path (str): Path to the training dataset.
            test_path (str): Path to the test dataset.
            store_path (str): Path to the store dataset.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
        self.train_data = None
        self.test_data = None
        self.store_data = None
        self.model_pipeline = None

        logging.info("StoreSalesPrediction class initialized.")

    def load_and_merge_data(self):
        """
        Load datasets and merge train/test data with store data.
        """
        logging.info("Loading data...")
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            self.store_data = pd.read_csv(self.store_path)

            logging.info(f"Train data loaded with shape: {self.train_data.shape}")
            logging.info(f"Test data loaded with shape: {self.test_data.shape}")
            logging.info(f"Store data loaded with shape: {self.store_data.shape}")

            # Merging store data with training and test data
            logging.info("Merging train and test data with store data...")
            self.train_data = self.train_data.merge(self.store_data, on="Store", how="left")
            self.test_data = self.test_data.merge(self.store_data, on="Store", how="left")
        except Exception as e:
            logging.error(f"Error loading and merging data: {e}")
            raise e

    def preprocess_data(self, data):
        """
        Preprocess data: handle datetime, missing values, and feature engineering.
        Args:
            data (pd.DataFrame): Data to preprocess.
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        logging.info("Preprocessing data...")

        # Handle datetime
        data['Date'] = pd.to_datetime(data['Date'])
        data['Weekday'] = data['Date'].dt.weekday
        data['IsWeekend'] = data['Weekday'].isin([5, 6]).astype(int)
        data['Month'] = data['Date'].dt.month
        data['Year'] = data['Date'].dt.year

        # Feature Engineering
        data['DaysToHoliday'] = data['Date'].apply(self._days_to_holiday)
        data['DaysAfterHoliday'] = data['Date'].apply(self._days_after_holiday)
        data['MonthPhase'] = data['Date'].dt.day.apply(self._month_phase)

        # Handle missing values
        data.fillna({
            'CompetitionDistance': data['CompetitionDistance'].median(),
            'Promo2SinceWeek': 0,
            'PromoInterval': 'Unknown'
        }, inplace=True)

        return data

    def _days_to_holiday(self, date):
        """Dummy function for days to holiday calculation."""
        return 0  # Replace with actual logic

    def _days_after_holiday(self, date):
        """Dummy function for days after holiday calculation."""
        return 0  # Replace with actual logic

    def _month_phase(self, day):
        """Categorize a day into beginning, middle, or end of the month."""
        if day <= 10:
            return 'Beginning'
        elif day <= 20:
            return 'Middle'
        else:
            return 'End'

    def prepare_features(self):
        """
        Preprocess training and test data.
        """
        logging.info("Preparing features...")
        self.train_data = self.preprocess_data(self.train_data)
        self.test_data = self.preprocess_data(self.test_data)

   