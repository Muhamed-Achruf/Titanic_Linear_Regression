# Titanic Linear Regression

## About
This is my first Machine Learning project. I built a Linear Regression model to predict the ticket fare (Fare) on the Titanic dataset.

## Steps Performed:
1. **Data Cleaning:** Handled missing values in `Age` (using Median) and `Embarked` (using Mode).
2. **Feature Engineering:** Converted categorical columns (`Sex`, `Embarked`) into numerical values.
3. **Handling Outliers:** Applied `Log1p` transformation on the `Fare` column, which significantly improved the model's performance.
4. **Model Training:** Used `LinearRegression` from Scikit-Learn.
5. **Evaluation:** Achieved an **R² Score of ~0.66**.

## Features Used:
- Age
- Sex
- Pclass (Ticket class)
- SibSp & Parch (Family members)
- Embarked (Port of embarkation)

## Libraries:
- Pandas, NumPy, Scikit-Learn, Matplotlib.
