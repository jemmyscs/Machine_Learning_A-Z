# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Importing the dataset
# dataset = pd.read_csv('Position_Salaries.csv')
dataset = pd.read_csv('em_stock_data.csv')
period = 10
X = dataset.loc[:, ['Close', 'Volume']].values
y = dataset.loc[:, ['Close']].values

X = X[0:len(X) - period]
y = y[period:len(y)]

# 2. Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split # cross_validation is deprecated，使用 model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle=False)

# 3. Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# 4. Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train, y_train)

# 5. Predicting a new result
# y_pred = regressor.predict([[183, 62303300]])
y_pred = regressor.predict(X_test)
print("predict value: ", y_pred)

# 6. Visualising the Random Forest Regression results (higher resolution)
# 进行反标准化
# y_pred_plot = y_pred.reshape(len(y_pred), 1) # 一维转为二维 （单列)
# y_pred_plot = sc_y.inverse_transform(y_pred_plot)
X_grid = dataset.loc[:, ['Num']].values[0:len(y_test)]
plt.scatter(X_grid, y_test, color = 'red')
plt.plot(X_grid, y_pred, color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()