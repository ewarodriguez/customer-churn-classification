# Customer Churn Prediction Model

## 📌 Project Overview
Customer churn—the rate at which customers close their accounts—is a critical metric for retail banking. Because retaining an existing customer is significantly more cost-effective than acquiring a new one, predicting churn allows financial institutions to take proactive retention measures.

The primary objective of this project is to build a predictive machine learning classification system. This system identifies high-risk customers before they leave, enabling targeted retention campaigns and personalized incentives to safeguard revenue.

## 📊 Dataset & Variables
This project utilizes a banking industry dataset sourced from Kaggle. 

* **Data Source:** (https://www.kaggle.com/datasets/harshitstark/dataset/data)
* **Target Variable:** `Exited` (Binary classification representing customer churn)

### Data Dictionary

| Variable Name | Description |
| :--- | :--- |
| **id** | Unique sequential row index. |
| **CustomerId** | Unique identifier for each customer. |
| **Surname** | The surname of the customer. |
| **CreditScore** | The customer's credit score. |
| **Geography** | The country or region of residence. |
| **Gender** | The gender of the customer. |
| **Age** | The age of the customer. |
| **Tenure** | Number of years the customer has been with the bank. |
| **Balance** | Current account balance. |
| **NumOfProducts** | Number of bank products/services utilized. |
| **HasCrCard** | Credit card ownership indicator (1 = Yes, 0 = No). |
| **IsActiveMember** | Active status indicator (1 = Yes, 0 = No). |
| **EstimatedSalary** | Estimated annual salary of the customer. |
| **Exited** | Churn status indicator (1 = Churned, 0 = Retained). |


## ⚙️ Methodology & Algorithms
To find the optimal predictive model, various supervised machine learning classification algorithms were implemented and evaluated:
* **Random Forest Classifier:** Implemented to handle non-linear relationships and feature interactions (saved model file: `RFClassifier_Model.pkl`).
* **XGBoost Classifier:** Implemented as a high-performance gradient boosting model to maximize predictive accuracy and capture complex interactions.

* *[Optional: Add other algorithms here if you tested them, e.g., Decision Trees, XGBoost, etc.]*

## 📁 Repository Structure
```text
├── streamlit/                   
│   └── config.toml                          # Customize UI themes
├── models/
│   └── RFClassifier_Model.pkl               # Trained Random Forest model file
│   └── XgBoostClassifier_Model.pkl          # Trained XGBoost model file
│   └── RobustScaler.pkl                     # Robust Scaler file
├── notebooks/
│   └── Classification_Customer_Churn.ipynb  # Jupyter notebook containing EDA and model building
├── .gitignore                               # Configuration to exclude data files and dependencies
└── app.py                                   # Main Python App File
└── requirements.txt                         # Requirements File containing the package needed to build the app
└── README.md                                # Project documentation
```

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com
cd customer-churn-classification
```

### 2. Set up your virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install required packages
Ensure you create a `requirements.txt` file listing packages like `scikit-learn`, `pandas`, `numpy`, and `jupyter`. Then run:
```bash
pip install -r requirements.txt
```

### 4. Explore the notebooks
Launch Jupyter Notebook to view the data analysis and training scripts:
```bash
jupyter notebook
```
