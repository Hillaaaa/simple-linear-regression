import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/study_hours.csv")

X = df[["hours_studied"]]
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print( predictions)
print(y_test)