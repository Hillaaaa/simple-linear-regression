import pandas as pd

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

study_df = load_data("data/study_hours.csv")
housing_df = load_data("data/housing.csv")

print(study_df.shape)
print(housing_df.shape)