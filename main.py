# All the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Loading credit card data

credit_card_data=pd.read_csv("creditcard.csv")
# data information
#print(credit_card_data.info())
# checking the mission values in data 
#print(credit_card_data.isnull().sum())
# distribution of legit transaction and fraudulent transaction
#print(credit_card_data["Class"].value_counts())
# data is unbalanced , 0= Legit Transaction ,1= Fraudulent transaction
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]
# print(legit.shape)
# print(fraud.shape)
# Statistical measure of the data
# print(legit.Amount.describe())
# print(fraud.Amount.describe())

# print(credit_card_data.groupby("Class").mean())
 # building sample data set from the original data set
  # number of fredulent transaction is 492

legit_sample=legit.sample(n=492)
new_dataset=pd.concat([legit_sample,fraud],axis=0)
new_dataset.head()
new_dataset.tail()
# print(new_dataset["Class"].value_counts())

# print(new_dataset.groupby("Class").mean())

#Spliting the data into different features and targets
X=new_dataset.drop(columns="Class",axis=1)
Y=new_dataset["Class"]
# print(X)
# print(Y)

# spliting data into training data and testing data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
#print(X.shape,x_train.shape,x_test.shape)


# Model training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000, solver='liblinear')
print(model.fit(X_train_scaled, y_train))

# model evaluation
# accuracy on training data
x_train_prediction=model.predict(X_train_scaled)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy score of training data",training_data_accuracy)

# Accuracy on test data
x_test_prediction=model.predict(X_test_scaled)
testing_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy score of testing data",testing_data_accuracy)