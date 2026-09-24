import pandas as pd

df = pd.read_csv("data/housing.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

median_bedrooms = df["total_bedrooms"].median()
df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)

print(df.isna().sum())

