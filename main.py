from src.pipeline import load_data, select_feature_target, train_model, split_data
df = load_data("data/study_hours.csv")
X, y = select_feature_target(df, "hours_studied", "exam_score")
X_train, X_test, y_train, y_test = split_data(X,y)
model = train_model(X_train, y_train)
print(model.coef_, model.intercept_)

df = load_data("data/housing.csv")
X, y = select_feature_target(df, "median_income", "median_house_value")
X_train, X_test, y_train, y_test = split_data(X,y)
model = train_model(X_train, y_train)
print(model.coef_, model.intercept_)
