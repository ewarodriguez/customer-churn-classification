# Customer Churn Prediction Model

## 📌 Project Overview
Customer churn—the rate at which customers cancel their subscriptions—is a vital metric for businesses offering subscription services. Because it is significantly less expensive to retain an existing customer than it is to acquire a new one, predicting churn allows companies to take proactive measures for customer retention.

The primary objective of this project is to build a predictive classification system that identifies high-risk customers before they churn, enabling targeted marketing campaigns and personalized incentives to improve business retention.

## 📊 Dataset & Variables
This project utilizes a dummy company dataset sourced from Kaggle. 
* **Data Source:** [Kaggle Customer Churn Dataset](https://kaggle.com)
* **Target Variable:** `Exited` (Binary classification representing customer churn).

### Data Dictionary

| Variable Name | Description |
| :--- | :--- |
| **id** | The sequential number assigned to each row in the dataset. |
| **CustomerId** | A unique identifier for each customer. |
| **Surname** | The surname of the customer. |
| **CreditScore** | The credit score of the customer. |
| **Geography** | The geographical location of the customer (e.g., country or region). |
| **Gender** | The gender of the customer. |
| **Age** | The age of the customer. |
| **Tenure** | The number of years the customer has been with the bank. |
| **Balance** | The account balance of the customer. |
| **NumOfProducts** | The number of bank products the customer has. |
| **HasCrCard** | Indicates whether the customer has a credit card (binary: yes/no). |
| **IsActiveMember** | Indicates whether the customer is an active member (binary: yes/no). |
| **EstimatedSalary** | The estimated salary of the customer. |
| **Exited** | Indicates whether the customer has exited the bank (binary: yes/no). |

## ⚙️ Methodology & Algorithms
To find the optimal predictive model, various supervised machine learning classification algorithms were implemented and evaluated:
* **Logistic Regression:** Used as a strong baseline model for linear feature relationships.
* **Random Forest Classifier:** Implemented to handle non-linear relationships and feature interactions (saved model file: `RFClassifier_Model.pkl`).
* *[Optional: Add other algorithms here if you tested them, e.g., Decision Trees, XGBoost, etc.]*

## 📁 Repository Structure
```text
├── models/
│   └── RFClassifier_Model.pkl   # Trained Random Forest model file
├── notebooks/
│   └── exploration_training.ipynb  # Jupyter notebook containing EDA and model building
├── .gitignore                   # Configuration to exclude data files and dependencies
└── README.md                    # Project documentation
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
