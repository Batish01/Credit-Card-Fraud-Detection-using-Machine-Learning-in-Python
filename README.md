# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using supervised machine learning techniques. It involves data preprocessing, model training, and performance evaluation using logistic regression.

## 📌 Objective

To build a predictive model that can accurately classify credit card transactions as fraudulent or genuine based on historical data.

## 📂 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains transactions made by European cardholders in September 2013.
- Highly imbalanced dataset:
  - Total Transactions: 284,807
  - Fraudulent: 492 (~0.172%)

## 🔧 Tools and Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 🧠 Model Used

- **Logistic Regression**
  - Solver: `liblinear`
  - Increased `max_iter` to ensure convergence

## ⚙️ Steps Performed

1. Data loading and exploration
2. Data preprocessing
   - Dropping irrelevant columns
   - Handling imbalanced data
3. Model training using logistic regression
4. Model evaluation

## 📈 Evaluation Metrics

- **Training Accuracy:** 94.79%
- **Testing Accuracy:** 92.89%

> Note: Accuracy is not the only metric for imbalanced datasets. Precision, recall, and F1-score were also considered.

## 📊 Model Performance

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

