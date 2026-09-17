import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/study_hours.csv")

print(df)
print(df.head())
print(df.shape)
print(df.columns)
df.info()
print(df.describe())
print(df.isnull().sum())
print(df["exam_score"])
print(df[df["hours_studied"] > 5])

plt.scatter(df['hours_studied'],df["exam_score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.show()

