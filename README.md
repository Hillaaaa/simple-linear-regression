## Simple Linear Regression Project

This is a project where I learned simple linear regression properly — not just calling a library function, but actually understanding the math behind it. I built it manually with NumPy first, then did the same thing with scikit-learn, and compared the two to make sure they matched.

After that I turned it into a reusable pipeline and ran it on three different datasets: a small handmade one (hours studied vs exam score), a real housing dataset, and a real advertising dataset. I explored and cleaned the data, trained models, checked them with a few different metrics (MAE, MSE, RMSE, R²), and compared the results to understand what those metrics actually tell you — and when a good-looking score shouldn't be trusted.

## What I Learned

- How to build a simple regression model manually using math and NumPy
- How to build the same model using scikit-learn
- How to write reusable functions instead of copy-pasting code for every dataset
- How to use Git: adding, committing files, and working with branches
- How to use Windows PowerShell — I started this project not really knowing the terminal at all
- How to evaluate a model using different metrics (MAE, RMSE, R²), and compared these across three datasets: a handcrafted dataset, a real housing dataset, and a real advertising dataset (TV spend vs sales)
- A bit of data exploration and cleaning — handling missing values and outliers

## Datasets

**Study Hours (handcrafted):** A small dataset I made myself — 10 rows, 2 columns. Feature: hours studied. Target: exam score. It's a near-perfect linear relationship, made for learning the basics before using real data.

**California Housing:** A real dataset from 1990 census data — 20,640 rows, 10 columns. Feature: median income. Target: median house value. I filtered out districts where the house value was capped at $500,000, since that was a data collection artifact, not a real value.

**Advertising:** A real, commonly-used dataset from the ISLR statistics textbook — 200 rows, 5 columns. Feature: TV ad spend. Target: Sales.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- Git & GitHub

## Project Structure

simple-linear-regression/
│
├── data/
│ ├── study_hours.csv
│ ├── housing.csv
│ └── advertising.csv
│
├── notebooks/
│ └── comparison.ipynb
│
├── src/
│ ├── explore_data.py
│ ├── manual_regression.py
│ ├── sklearn_regression.py
│ ├── train_test_regression.py
│ ├── handling_missing.py
│ └── pipeline.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md


## How to Run

git clone https://github.com/Hillaaaa/simple-linear-regression.git
cd simple-linear-regression
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py


## Methodology

For each dataset, I followed the same process: loaded the data, explored it, cleaned it (handling missing values and outliers where needed), then selected the feature and target columns. I split the data into training and test sets, trained the model on the training data only, and tested it on the unseen test data. Finally, I evaluated the model using several metrics and plotted graphs to visualize the results.

## Model

I used the `LinearRegression` model from scikit-learn. But prior to that, I tried to build the model manually with NumPy, to understand the math behind it. I compared the two and confirmed that they matched.

## Metrics

I evaluated my models using four metrics:

- **MAE (Mean Absolute Error):** the absolute value of each residual, averaged — tells you, on average, how far off the model's predictions are from the real values.
- **MSE (Mean Squared Error):** squares the errors before averaging them, which punishes large errors (outliers) more heavily than small ones.
- **RMSE (Root Mean Squared Error):** the square root of MSE, which brings the error back into the same units as the target, making it easier to interpret than MSE.
- **R² (R-squared):** measures how much of the variation in the target variable the model explains — closer to 1 means a better fit, but it needs to be read alongside sample size, since a small test set can make R² misleadingly high.

## Results

| Dataset | MAE | RMSE | R² |
|---|---|---|---|
| Study Hours | 0.30 | 0.39 | 1.00 |
| Housing | 57,910.58 | 76,059.47 | 0.42 |
| Advertising | 2.44 | 3.19 | 0.68 |

Study Hours shows an R² of 1.00 because the data was handcrafted, so the feature mapped onto the target almost perfectly — it's not a sign of a genuinely strong model. You can't directly compare Housing's MAE to the other datasets, since its units are in dollars, running into the tens of thousands, unlike exam points or sales figures. Out of the three, I'd trust the Advertising model the most — it's real data, with a reasonably sized sample, and a solid R² without any artificial perfection behind it.

## Visualizations

- **Scatter plot with regression line:** shows the actual data points against the fitted regression line, to visually check how well the model fits.
- **Actual vs. Predicted:** compares the model's predicted values against the real values — points closer to the diagonal line mean better predictions.
- **Residual plot:** shows how far off each prediction is from the actual value, which helps spot patterns in the model's errors (like the systematic bias I found in the Housing model).

## Limitations

- The Study Hours dataset was too small and handcrafted to be perfect, so it doesn't give any valuable real-world insight — it was mainly useful for learning the basics.
- The Housing model could not be fully explained using only one feature (income) — house value depends on much more than that.
- In the Advertising dataset, we only used TV spend and ignored Radio and Newspaper, which might have held valuable information.
- We didn't do any cross-validation — results are based on a single train/test split, which could shift somewhat with a different split.
- We only tried the `LinearRegression` model and didn't test any other model types.

## Future Improvements

- Use more real-world datasets instead of handcrafted ones, to get more meaningful, trustworthy results.
- Try multiple linear regression (using more than one feature) to reduce underfitting, especially on the Housing dataset.
- Test other model types to see how they compare against `LinearRegression`.
- Add cross-validation instead of relying on a single train/test split.