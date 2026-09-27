import pandas as pd
from sklearn.linear_model import LinearRegression

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