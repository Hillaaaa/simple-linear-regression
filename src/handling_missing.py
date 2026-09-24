from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/housing.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

median_bedrooms = df["total_bedrooms"].median()
df["total_bedrooms"] = df["total_bedrooms"].fillna(median_bedrooms)

print(df.isna().sum())

print(df["ocean_proximity"].unique())
print(df["ocean_proximity"].value_counts())

df = df[["median_income", "median_house_value"]]
print(df.head())
print(df.shape)
df = df[df["median_house_value"]<500001]
print(df.shape)

X = df[["median_income"]]
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,predictions)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print(df["median_house_value"].describe())

plt.hist(df["median_house_value"], bins=50)
plt.xlabel("Median House Value")
plt.ylabel("Number of Districts")
plt.title("Distribution of Median House Values")
plt.show()
