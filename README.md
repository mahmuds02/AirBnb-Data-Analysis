# 🏠 AirBnB Data Analysis & Price Prediction

A comprehensive data science project involving extensive data cleaning, exploratory data analysis, feature engineering, and machine learning models to analyze and predict AirBnB listing prices.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tableau](https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Project Overview

This project analyzes AirBnB listing data to understand pricing patterns and build predictive models. The workflow includes comprehensive data cleaning of 58+ features, exploratory data analysis, feature engineering, and comparison of multiple machine learning algorithms with and without PCA dimensionality reduction.

## 🎯 Key Objectives

- Clean and preprocess raw AirBnB listing data
- Perform exploratory data analysis to uncover insights
- Engineer meaningful features for prediction
- Build and compare multiple regression models
- Visualize findings using Tableau dashboards

## 📁 Repository Structure

```
├── AirBnB.ipynb                    # Main Jupyter notebook with all analysis
├── AirBnB.xlsx                     # Source dataset
├── Presentation.pptx               # Project presentation
├── README.md                       # Project documentation
└── Tableau Dashboard               # Interactive visualizations (Tableau Public)
```

## 🧹 Data Cleaning Process

Extensive cleaning was performed on **58 columns** including:

| Category | Columns Cleaned |
|----------|-----------------|
| **Location** | City, State, Country, Zipcode, Host Location, Latitude, Longitude |
| **Host Info** | Host ID, Host Name, Host About, Host Since, Host Verifications |
| **Property Details** | Room Type, Bed Type, Property Type, Bedrooms, Bathrooms, Beds, Square Feet |
| **Pricing** | Price, Security Deposit, Cleaning Fee, Extra People |
| **Reviews** | Number of Reviews, Reviews per Month, All Review Scores (7 categories) |
| **Availability** | Availability 30/60/90/365 days |
| **Policies** | Cancellation Policy, House Rules, Minimum/Maximum Nights |
| **Amenities** | Features, Amenities, Experiences Offered |

### Cleaning Techniques Applied:
- Handling missing values with intelligent imputation
- Standardizing text fields (City names, State names)
- Converting data types (dates, currencies, percentages)
- Removing duplicates and irrelevant columns
- Creating zipcode mappings for 82+ cities
- Encoding categorical variables

## 📊 Exploratory Data Analysis

- **Correlation Analysis** — Identifying relationships between features
- **Price Distribution** — Understanding pricing patterns across locations
- **Feature Importance** — Determining key factors affecting price
- **Geographic Analysis** — Mapping listings across countries and cities

## 🔧 Feature Engineering

- **Profit Calculation** — Derived metric for business insights
- **One-Hot Encoding** — Categorical variable transformation
- **Standard Scaling** — Normalizing numerical features
- **PCA (Principal Component Analysis)** — Dimensionality reduction

## 🤖 Machine Learning Models

Multiple regression models were trained and evaluated with and without PCA:

| Model | With PCA | Without PCA |
|-------|----------|-------------|
| **Linear Regression** | ✅ | ✅ |
| **Lasso Regression** | ✅ | ✅ |
| **Ridge Regression** | ✅ | ✅ |
| **Random Forest** | ✅ | ✅ |

### Evaluation Metrics:
- R² Score (Coefficient of Determination)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## 📈 Visualizations

Interactive Tableau dashboards showcasing:
- Price distribution by location
- Host performance metrics
- Amenity impact on pricing
- Seasonal availability patterns
- Review score correlations

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Programming** | Python 3.x |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Tableau |
| **Machine Learning** | Scikit-learn |
| **Development** | Jupyter Notebook |

## 📚 Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
```

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/mahmuds02/AirBnB-Data-Analysis.git
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn word2number unidecode tqdm
   ```

3. **Open Jupyter Notebook**
   ```bash
   jupyter notebook AirBnB.ipynb
   ```

4. **Run all cells** to reproduce the analysis

## 📊 Key Findings

- Location significantly impacts listing prices
- Amenities and property type are strong price predictors
- Review scores correlate with booking success
- Random Forest outperformed linear models in prediction accuracy

## 👤 Author

**Saim Mahmud**  
Data Science & Analytics Graduate Student  
Buffalo State University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saimmahmud/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mahmuds02)

## 📄 License

This project is for educational purposes as part of coursework at Buffalo State University.

---

*Fall 2025| Data Science Course Project*
