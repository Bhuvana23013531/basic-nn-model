# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The term neural network refers to a group of interconnected units called neurons that send signals to each other. While individual neurons are simple, many of them together in a network can perform complex tasks. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression is a method for understanding the relationship between independent variables or features and a dependent variable or outcome. Outcomes can then be predicted once the relationship between independent and dependent variables has been estimated.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 4 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![Screenshot 2024-02-19 193218](https://github.com/Bhuvana23013531/basic-nn-model/assets/147125678/d22bd0ce-3419-4e5d-bb53-b99515605799)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:BHUVANESHWARI M
### Register Number:212223230033
```
Name:BHUVANESHWARI M
Register Number:212223230033

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import defaultauth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('deeplearn').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'input':'float'})
dataset1 = dataset1.astype({'output':'float'})
dataset1.head(10)
X = dataset1[['input']].values
y = dataset1[['output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai = Sequential([
    Dense(units= 4, activation = 'relu' ,input_shape = [1]),
    Dense(units = 1)
])
ai.compile(optimizer="rmsprop",loss='mse')
ai.fit(X_train1, y_train,epochs=3000)
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)
X_n1 = [[10]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)

```
## Dataset Information

![Screenshot 2024-02-19 161717](https://github.com/Bhuvana23013531/basic-nn-model/assets/147125678/e762120c-79e4-42f8-ab33-e0f2f5f0f7e8)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-02-19 194520](https://github.com/Bhuvana23013531/basic-nn-model/assets/147125678/3c1e4db2-070a-47ef-b72f-8528f3495dda)


### Test Data Root Mean Squared Error

![Screenshot 2024-02-19 194732](https://github.com/Bhuvana23013531/basic-nn-model/assets/147125678/44a9c671-da71-46fd-96ff-d17f1d592a59)


### New Sample Data Prediction

![Screenshot 2024-02-19 194629](https://github.com/Bhuvana23013531/basic-nn-model/assets/147125678/fa1744e0-678c-4f68-9eb6-a9d1c10c3ffe)


## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.
