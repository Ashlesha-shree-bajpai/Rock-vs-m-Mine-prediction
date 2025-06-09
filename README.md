# Rock-vs-m-Mine-prediction
This is a basic machine learning project based on the sonar data with the help of pandas numpy and sklearn, it uses logistic regression model which is best for binary classification problems. It helps to determine whether the submarine has detected a rock or a mine.


 CODE:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Data collection and Data processing

# loading the dataset to pandas dataframe
sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header=None)


sonar_data.head()


# number of rows and colms
sonar_data.shape

sonar_data.describe()

sonar_data[60].value_counts()

M --> Mine

R --> Rock

sonar_data.groupby(60).mean()

# separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

print(X_train)
print(Y_train)

Training and Test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

Model Training --> Logistic Regression

model = LogisticRegression()

# training the logistic regression model with training data
model.fit(X_train, Y_train)


Model Evaluation

#accuracy on training data(any accuracy greater than 70% is good)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data(any accuracy greater than 70% is good)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


print('Accuracy on test data :', test_data_accuracy)

Making a Predictive System

input_data = (0.0310,0.0221,0.0433,0.0191,0.0964,0.1827,0.1106,0.1702,0.2804,0.4432,0.5222,0.5611,0.5379,0.4048,0.2245,0.1784,0.2297,0.2720,0.5209,0.6898,0.8202,0.8780,0.7600,0.7616,0.7152,0.7288,0.8686,0.9509,0.8348,0.5730,0.4363,0.4289,0.4240,0.3156,0.1287,0.1477,0.2062,0.2400,0.5173,0.5168,0.1491,0.2407,0.3415,0.4494,0.4624,0.2001,0.0775,0.1232,0.0783,0.0089,0.0249,0.0204,0.0059,0.0053,0.0079,0.0037,0.0015,0.0056,0.0067,0.0054)

#changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'): #the first ele in the list
  print('The object is a rock')
else:
  print('The object is mine')

