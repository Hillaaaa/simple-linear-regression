import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        df = pd.read_json(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("unsupported file type: " + file_path)
    return df


def select_feature_target(df, feature_column, target_column):
    X = df[[feature_column]]
    y = df[target_column]
    return X, y

def train_model(X,y):
    model = LinearRegression()
    model.fit(X,y)
    return model

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
def evaluate_model(model, X_test, y_test):
    predictions= model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2
    }
    return metrics
def prepare_data(df, feature_column, target_column):
    cleaned_df = df.dropna(subset=[feature_column, target_column])
    return cleaned_df

def plot_results(X, y, model, xlabel, ylabel, title):
    X_values = X.iloc[:,0]
    X_sorted = X.sort_values(by=X.columns[0])
    plt.figure(figsize=(10,6))
    plt.scatter(X_values, y, color="blue", label="Actual data")
    plt.plot(X_sorted.iloc[:,0], model.predict(X_sorted),color= "black", label= "Regression line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
