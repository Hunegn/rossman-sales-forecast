import pandas as pd
import numpy as np
import logging

class DataCleaner:
    def __init__(self, train_path, store_path, output_path):
        """
        Initialize the DataCleaner class.
        Args:
            train_path (str): Path to the training dataset.
            store_path (str): Path to the store dataset.
            output_path (str): Path to save the cleaned dataset.
        """
        self.train_path = train_path
        self.store_path = store_path
        self.output_path = output_path
        self.train_data = None
        self.store_data = None
        self.cleaned_data = None

        logging.basicConfig(
            filename="data_cleaning.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("DataCleaner initialized.")

    def load_data(self):
        """
        Load the train and store datasets.
        """
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.store_data = pd.read_csv(self.store_path)
            logging.info("Datasets loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading datasets: {e}")
            raise e

    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        """
        logging.info("Handling missing values...")
        numeric_cols = self.cleaned_data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.cleaned_data.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            self.cleaned_data[col] = self.cleaned_data[col].fillna(self.cleaned_data[col].median())

        for col in categorical_cols:
            self.cleaned_data[col] = self.cleaned_data[col].fillna("Unknown")

    def detect_and_handle_outliers(self):
        """
        Detect and handle outliers using the IQR method.
        """
        logging.info("Detecting and handling outliers...")
        numeric_cols = self.cleaned_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            Q1 = self.cleaned_data[col].quantile(0.25)
            Q3 = self.cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.cleaned_data[col] = np.where(
                (self.cleaned_data[col] < lower_bound) | (self.cleaned_data[col] > upper_bound),
                np.nan,  
                self.cleaned_data[col]
            )
        self.handle_missing_values()

    def save_cleaned_data(self):
        """
        Save the cleaned data to a CSV file.
        """
        try:
            self.cleaned_data.to_csv(self.output_path, index=False)
            logging.info(f"Cleaned data saved to {self.output_path}.")
        except Exception as e:
            logging.error(f"Error saving cleaned data: {e}")
            raise e

    def run_cleaning_pipeline(self):
        """
        Run the complete data cleaning pipeline.
        """
        self.load_data()
        self.cleaned_data = self.train_data.merge(self.store_data, on="Store", how="left")
        self.handle_missing_values()
        self.detect_and_handle_outliers()
        self.save_cleaned_data()

if __name__ == "__main__":
    cleaner = DataCleaner(
        train_path="../data/train.csv",
        store_path="../data/store.csv",
        output_path="../data/cleaned_data.csv"
    )
    cleaner.run_cleaning_pipeline()
