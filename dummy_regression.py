from sklearn.dummy import DummyRegressor
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# assign csv file to variable
csv = pandas.read_csv("C:\Biorobotics\Introduction to Artificial Intelligence\Coursework\code\coursework_other.csv")

# assigns the feature space to variable x (drops the target to leave all the variables)
x = csv.drop('PE', axis = 1)
# assigns the target variable to y
y = csv['PE']

# split data into training and testing, test size of 0.2 (80:20 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 69)

# create dummy 
dummy = DummyRegressor(strategy = 'mean')

# train dummy
dummy.fit(x_train, y_train)

# make prediction
y_prediction = dummy.predict(x_test)

# generate a mean squared error 
mse =  mean_squared_error(y_test, y_prediction)

print("Mean Squared Error (Test): {:.2f}".format(mse))

#plotting the data
plt.scatter(y_test, y_prediction, color = 'blue', alpha = 0.5, s = 1, label = "Dummy Regression Prediction")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--',  label = 'Perfect Scenario')
plt.xlabel('Actual PE')
plt.ylabel('Predicted PE')
plt.title('Predicted vs Actual PE')
plt.legend()
plt.show()
