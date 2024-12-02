import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("formattedcryptodata2.csv")

X = df[['VIX', 'SOFR', 'Inflation', 'Return', 'UR']]
y = df[['Conservative','Balanced','Growth']]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 222)

dec = [13.51,4.5,3.2,0.03,4]
jan = [15,4.4,3.5,0.025,3.5]

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

residuals_conservative = y_test['Conservative'] - y_pred[:, 0]
residuals_balanced = y_test['Balanced'] - y_pred[:, 1]
residuals_growth = y_test['Growth'] - y_pred[:, 2]

res_con = np.array(residuals_conservative)
res_bal = np.array(residuals_balanced)
res_growth = np.array(residuals_growth)

dec_pred = regr.predict([dec])
jan_pred = regr.predict([jan])

# Compare actual and predicted values
print("Actual values:\n", y_test)
print("\nPredicted values:\n", y_pred)
print("Base values\n", np.array(y['Conservative']))
print("\n", np.array(y['Balanced']))
print("\n", np.array(y['Growth']))
print("Test\n", y_pred[:, 0])
print("\n", y_pred[:, 1])
print("\n", y_pred[:, 2])
print("Residuals\n", res_con)
print("\n", res_bal)
print("\n", res_growth)
print("Predicted\n", dec_pred)
print("\n", jan_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
overall_mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error for each output:", mse)
print("Overall Mean Squared Error:", overall_mse)

print(regr.coef_)