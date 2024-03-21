# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
1. Use the standard libraries in python for Gradient Design.
2. Upload the dataset and check any null value using .isnull() function.
3. Declare the default values for linear regression.
4. Calculate the loss usinng Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using scatter plot function.


## Program:
````
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hoshini S 
RegisterNumber:  2305003006
*/
````
````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
dh.head()

df.tail()

#segregating data to variables
x = df.iloc[:,:-1].values
x

y = df.iloc[:,1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual value
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="magenta")
plt.plot(x_train,regressor.predict(x_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,regressor.predict(x_test),color="gray")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse = np.sqrt(mse)
print('RMSE',rmse)


```````
## Output
df.head():

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/62590047-f712-49f5-a103-99bd976f8841)

df.tail():

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/3d895f25-41d0-493e-b640-4ff89f96a1af)

x values:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/4b6ea6c9-13bb-4551-9b6f-f0ae549fe3fa)

y values:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/f4771238-c788-4062-8e23-99879410e3cd)

y_pred:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/5f7ea461-dcb0-4a3d-b70e-f40862af3036)

y_test:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/b82a492f-77a0-48be-a8e6-6a98e0585dc0)

Graph of training data:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/b7bb2f54-af01-49a2-b13d-b6e9848fa802)

Graph of test data:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/185cd765-57f0-424b-bf6e-8793789dea8d)

Values of MSE, MAE, RMSE:

![image](https://github.com/Revathi-Dayalan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/96000574/e9affb7a-53d8-474b-9651-5ba19ac64820)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
