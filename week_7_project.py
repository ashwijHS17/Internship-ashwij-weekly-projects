# -*- coding: utf-8 -*-
"""Week 7 Project.ipynb
Original file is located at
    https://colab.research.google.com/drive/1ZVbadaCoS8y9ysP2RMNohG7uZV7T-8IL
"""

# 1. Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')


# 2. Load the Dataset
df = pd.read_csv('house_price_regression_dataset.csv')

# 3. Check Dataset

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# 4. Handle Missing Values

print("\nMissing values before:")
print(df.isnull().sum())

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after:")
print(df.isnull().sum())

# 5. Remove Duplicate Records

print("Duplicates before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates after:", df.duplicated().sum())

# 6. Perform Univariate Analysis

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# 7. Perform Bivariate Analysis

features = [col for col in numerical_cols if col != 'House_Price']

plt.figure(figsize=(15, 10))
for i, col in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=df[col], y=df['House_Price'])
    plt.title(f"{col} vs House Price")
plt.tight_layout()
plt.show()

# 8. Generate Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 9. Detect and Handle Outliers

def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])
    return df

for col in features:
    df = cap_outliers(df, col)

# 10. Apply Encoding (if needed)

categorical_cols = df.select_dtypes(include=['object']).columns

if len(categorical_cols) > 0:
    print("Applying One-Hot Encoding...")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    print("No categorical columns found.")

# 11. Apply Log Transformation
df['Log_House_Price'] = np.log1p(df['House_Price'])

# 12. Separate Features & Target

X = df.drop(['House_Price', 'Log_House_Price'], axis=1)
y = df['House_Price']

# 13. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 14. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 15. Train Regression Models

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# KNN Regression
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# 16. Evaluate Models


def evaluate(y_true, y_pred, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

    return mae, mse, rmse, mape, r2, adj_r2


# Linear Regression Evaluation
lr_results = evaluate(y_test, lr_pred, X.shape[1])

print("\nLinear Regression Results:")
print("MAE:", lr_results[0])
print("MSE:", lr_results[1])
print("RMSE:", lr_results[2])
print("MAPE:", lr_results[3])
print("R2 Score:", lr_results[4])
print("Adjusted R2:", lr_results[5])


# KNN Evaluation
knn_results = evaluate(y_test, knn_pred, X.shape[1])

print("\nKNN Regression Results:")
print("MAE:", knn_results[0])
print("MSE:", knn_results[1])
print("RMSE:", knn_results[2])
print("MAPE:", knn_results[3])
print("R2 Score:", knn_results[4])
print("Adjusted R2:", knn_results[5])
