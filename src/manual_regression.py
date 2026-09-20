import pandas as pd
import numpy as np

df = pd.read_csv("data/study_hours.csv")
x = df["hours_studied"].to_numpy()
y = df["exam_score"].to_numpy()

mean_x = np.mean(x)
mean_y = np.mean(y)

numerator = np.sum((x - mean_x)*(y-mean_y))
denominator = np.sum((x - mean_x)**2)
b1 = numerator/denominator
b0 = mean_y - (b1*mean_x)

predictions = b0 + b1*x

print(x)
print(y)
print(mean_x)
print(mean_y)
print(b1)
print(b0)
print(predictions)
