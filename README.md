# Rossmann Sales Prediction API

This repository contains a project that predicts daily sales for Rossmann stores using a machine learning model. It provides a REST API for real-time predictions and includes scripts for data preprocessing, model training, evaluation, and deployment.

---

## **Project Overview**

Rossmann is a chain of drugstores, and this project aims to predict daily sales across its various stores. Accurate sales predictions enable better decision-making and resource allocation, including inventory management, staffing, and promotional activities. The project follows these main tasks:

1. **Data Cleaning and Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering and Model Building**
4. **Model Evaluation and Interpretation**
5. **Model Serialization and Serving with FastAPI**
6. **Deployment for Real-Time Predictions**

---

## **Features**

### 1. Data Cleaning and Preprocessing
- Handle missing values and outliers.
- Create new features, such as weekday, weekends, and holiday proximity.
- Scale numerical features and encode categorical variables.

### 2. Exploratory Data Analysis
- Analyze sales trends and correlations.
- Visualize customer purchasing behavior.
- Investigate the impact of promotions and competitors.

### 3. Machine Learning Pipeline
- Implement a pipeline using scikit-learn for preprocessing and modeling.
- Train a Random Forest Regressor model to predict daily sales.

### 4. Deep Learning
- Build a Long Short-Term Memory (LSTM) model for time series sales prediction.
- Use TensorFlow/Keras for deep learning modeling.

### 5. API for Real-Time Predictions
- Serve the trained models using a REST API built with FastAPI.
- Provide endpoints for making predictions based on store features.

### 6. Deployment
- Deploy the API to a cloud platform for production use.
- Use tools like DVC for model tracking and Git for version control.

---

## **Repository Structure**

```
project/
├── data/                       # Contains raw and cleaned datasets
│   ├── train.csv
│   ├── test.csv
│   ├── store.csv
├── models/                     # Serialized models
│   └── sales_model.pkl
├── notebooks/                  # Jupyter notebooks for EDA and experiments
│   ├── data_cleaning.ipynb
│   ├── eda.ipynb
│   ├── modeling.ipynb
├── scripts/                    # Python scripts for tasks
│   ├── cleaning.py             # Data cleaning and preprocessing
│   ├── eda.py                  # Exploratory data analysis
│   ├── modeling.py             # Model training and evaluation
│   ├── deep_learning.py        # LSTM modeling
│   ├── api.py                  # FastAPI implementation
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── Procfile                    # Deployment configuration for Heroku
└── dvc.yaml                    # DVC pipeline configuration
```

---

## **Setup Instructions**

### Prerequisites

- Python 3.8+
- Virtual environment (optional but recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/Hunegn/rossmann-sales-prediction.git
cd rossmann-sales-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Data Preparation
Place the datasets (`train.csv`, `test.csv`, `store.csv`) in the `data/` directory.

### 5. Run the API Locally
```bash
uvicorn scripts.api:app --reload
```
Visit `http://127.0.0.1:8000` for the API health check.

---

## **Usage**

### Making Predictions
Send a POST request to `/predict/` with the following JSON body:
```json
{
  "CompetitionDistance": 500.0,
  "Promo": 1,
  "Weekday": 3,
  "DaysToHoliday": 10,
  "StoreType": "a",
  "Assortment": "basic",
  "MonthPhase": "Beginning"
}
```
Response:
```json
{
  "prediction": 12345.67
}
```

---

## **Future Work**

- Optimize model hyperparameters for better performance.
- Explore other algorithms like XGBoost or LightGBM.
- Improve deep learning model with more advanced architectures.
- Add authentication to the API for secure access.

---

## **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## **License**

This project is licensed under the MIT License.

