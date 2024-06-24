import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    data = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]
    data = data.dropna()
    X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
    y = data['SalePrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.show()

if __name__ == "__main__":
    filepath = 'data/house-prices-advanced-regression-techniques/train.csv'
    df = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
