import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

housing = fetch_california_housing()
df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)
df["MedHouseVal"] = housing.target
print("First 5 rows:")
print(df.head())
# Simple Linear Regression
X = df[["MedInc"]]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred_slr = slr.predict(X_test)

mae_slr = mean_absolute_error(y_test, y_pred_slr)
mse_slr = mean_squared_error(y_test, y_pred_slr)
rmse_slr = mse_slr ** 0.5
r2_slr = r2_score(y_test, y_pred_slr)
print("\nSimple Linear Regression")
print("MAE :", mae_slr)
print("MSE :", mse_slr)
print("RMSE:", rmse_slr)
print("R2  :", r2_slr)

#Multiple Linear Regression
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred_mlr = mlr.predict(X_test)

mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
mse_mlr = mean_squared_error(y_test, y_pred_mlr)
rmse_mlr = mse_mlr ** 0.5
r2_mlr = r2_score(y_test, y_pred_mlr)
print("\nMultiple Linear Regression")
print("MAE :", mae_mlr)
print("MSE :", mse_mlr)
print("RMSE:", rmse_mlr)
print("R2  :", r2_mlr)
results = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],
    "MAE": [mae_slr, mae_mlr],
    "MSE": [mse_slr, mse_mlr],
    "RMSE": [rmse_slr, rmse_mlr],
    "R2 Score": [r2_slr, r2_mlr]
})

print("\nModel Comparison:")
print(results)

plt.figure(figsize=(8, 5))
plt.scatter(X_test["MedInc"], y_test, alpha=0.5)
plt.scatter(X_test["MedInc"], y_pred_mlr, alpha=0.5)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("California Housing - Regression Predictions")

plt.show()