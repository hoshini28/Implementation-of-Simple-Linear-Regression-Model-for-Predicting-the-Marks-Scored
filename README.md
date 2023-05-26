# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard Libraries.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph

5. Predict the regression for marks by using the representation of the graph.

6. Compare the graphs and hence we obtained the linear regression for the given data

## Program:
````
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
X
Y = dataset.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="pink")
plt.plot(X_train,regressor.predict(X_train),color="orange") 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()



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
