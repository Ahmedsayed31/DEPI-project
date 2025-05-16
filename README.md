# Sales Forecasting and Demand prediction 

## Team Members:
- Ahemd Al Sayed Abdelrahman &nbsp;&nbsp;&nbsp; **Team Leader**
- Rehab Al Sayed Hassan 
- Mohamed Hisham Ahmed 
- Sherif Al Sayed Abadah  


# 🧠 Sales Forecasting ML Project

A machine learning project focused on forecasting future sales using historical data. The project incorporates robust feature engineering, model experimentation, version control, and experiment tracking with MLflow to enable production-ready deployment.

---

## 📊 Project Overview

This project aims to predict product sales using a range of features extracted from historical transactional data. The final model helps businesses make data-driven decisions regarding inventory management, demand planning, and marketing strategies.

---


---

## ⚙️ Tools & Technologies

- **Python** (Pandas, NumPy, Scikit-learn, XGBoost, CatBoost)
- **Optuna** for hyperparameter tuning
- **Matplotlib / Seaborn / Plotly** for data visualization
- **MLflow** for experiment tracking and model registry
- **Jupyter Notebooks** for exploration and development
- **VS Code / Git** for version control and collaboration

---

## 📈 Features Used

The dataset includes features like:

- `order_date` (engineered into: `day_of_week`, `month`, `week_of_year`)
- `segment`, `city`, `state`, `region`
- `category`, `sub_category`
- `discount`, `price_per_unit`, `quantity`, `profit`
- `time_taken_to_ship`
- Lag features: `sales_lag_1`, `sales_lag_2`, `sales_lag_3`

---

## 🔍 Model Development

- Handled missing values and outliers
- Feature engineering including lag variables and date decomposition
- Tried multiple models:
  - XGBoost
  - CatBoost
  - SARIMAX
  - Prophet
- Used **Optuna** for hyperparameter tuning
- Logged all experiments with **MLflow**
- Evaluated using:
  - RMSE
  - MAE
  - R² Score
  - Visual comparison with line charts (actual vs predicted)

---

## 🔬 Experiment Tracking (MLflow)

Tracked all experiments using MLflow, including:

- Parameters
- Metrics (RMSE, R², MAE, etc.)
- Model versions
- Feature importances
- Visualization artifacts

---

## 🚀 Next Steps / Production

- Register best model to MLflow Model Registry
- Serve model using `mlflow models serve`
- Develop a Flask or FastAPI API for real-time inference
- Create a cron job to retrain model weekly with fresh data
- Use Docker for deployment and reproducibility

---

## 📌 How to Run

1. Clone the repo
2. Install dependencies  
   ```bash
   pip install -r requirements.txt


