import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("data/study_hours.csv")

X = df[["hours_studied"]]
y = df["exam_score"]

model = LinearRegression()
model.fit(X,y)

predictions = model.predict(X)

print(model.coef_)
print(model.intercept_)
print(predictions)

plt.scatter(df["hours_studied"], df["exam_score"], color="black", label="Actual Data")
plt.plot(df["hours_studied"],predictions, color="red",label="Regression line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression Fit")
plt.legend()
plt.show()
