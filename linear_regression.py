import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# assign csv file to variable
csv = pandas.read_csv("C:\Biorobotics\Introduction to Artificial intelligence\Coursework\code\coursework_other.csv")

# assigns the feature space to variable x (drops the target to leave all the variables)
x = csv.drop('PE', axis = 1)
# assigns the target variable to y
y = csv['PE']

# split data into training and testing, test size of 0.2 (80:20 split) and val set
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size = 0.2, random_state = 69)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=69)

# create linear regression model
linear_regression = LinearRegression()

# train the model and predict the validation set
linear_regression.fit(x_train, y_train)
prediction = linear_regression.predict(x_val)

# calculate mse and r^2 values for prediction of validation set
mean_squared_error_validation = mean_squared_error(prediction, y_val)
r2_score_validation = r2_score(prediction, y_val)
print(f'mean squared error =', mean_squared_error_validation)
print(f'R^2 =', r2_score_validation)

