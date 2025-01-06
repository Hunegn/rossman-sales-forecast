import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

class CustomerBehaviorEDA:
    def __init__(self, cleaned_data_path):
        """
        Initialize the CustomerBehaviorEDA class.
        Args:
            cleaned_data_path (str): Path to the cleaned dataset.
        """
        self.cleaned_data_path = cleaned_data_path
        self.data = None

        
        logging.basicConfig(
            filename="eda.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("CustomerBehaviorEDA initialized.")

    def load_cleaned_data(self):
        """
        Load the cleaned data.
        """
        try:
            self.data = pd.read_csv(self.cleaned_data_path)
            logging.info("Cleaned data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading cleaned data: {e}")
            raise e

    def analyze_distributions(self):
        """
        Plot distributions of key features.
        """
        logging.info("Analyzing distributions...")
        for feature in ['Sales', 'Customers', 'Promo']:
            if feature in self.data.columns:
                sns.histplot(self.data[feature], kde=True, bins=30)
                plt.title(f"Distribution of {feature}")
                plt.xlabel(feature)
                plt.ylabel("Frequency")
                plt.show()

    def analyze_correlation(self):
        """
        Analyze and visualize the correlation matrix for numerical columns.
        """
        print("Analyzing correlation matrix...")

        
        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            print("No numerical columns found for correlation analysis.")
            return

        
        correlation_matrix = numeric_data.corr()

        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
        plt.title("Correlation Matrix")
        plt.show()


    def promo_effect_analysis(self):
        """
        Analyze the effect of promotions on sales.
        """
        logging.info("Analyzing promo effect on sales...")
        promo_sales = self.data[self.data['Promo'] == 1]['Sales']
        no_promo_sales = self.data[self.data['Promo'] == 0]['Sales']
        sns.boxplot(data=[promo_sales, no_promo_sales], notch=True)
        plt.xticks([0, 1], ['Promo', 'No Promo'])
        plt.ylabel('Sales')
        plt.title('Sales During Promotions vs. No Promotions')
        plt.show()

    def run_eda(self):
        """
        Run all EDA steps.
        """
        self.load_cleaned_data()
        self.analyze_distributions()
        self.analyze_correlation()
        self.promo_effect_analysis()


