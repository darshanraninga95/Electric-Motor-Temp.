import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import requests
import json# Importing the dataset
dataset = pd.read_csv("C:\\Users\\dell\\Desktop\\Project\\motor_temp.csv")
dataset.drop(['profile_id'],axis =1,inplace=True)
y=dataset["pm"].values
dataset.drop(["pm"],axis=1,inplace=True)
dataset.drop(["S.No"],axis=1,inplace=True)
predictors=list (dataset.columns)
x=dataset[predictors].values
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
regressor=RandomForestRegressor(n_estimators=50,n_jobs=-1) # we can change n_estimators with 100 or any number but took 10 for small pkl file.
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
import pickle
with open('model2.pkl','wb') as file:
    pickle.dump(regressor, file)

