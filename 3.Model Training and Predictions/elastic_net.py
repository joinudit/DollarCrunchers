# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split,KFold
def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

data_path='Data'
df=readExcel(data_path+'/output_final.xlsx','Sheet1')

y=(df.cad_us).values
#remove Date column and Predicted Column (CAD-USD)
df=df.drop("date",axis=1).drop("cad_us",axis=1)
X=df.values
#list of features
features=list(df.columns.values)
#Normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#Divide the data into train and test
X_train =  X[:6792]
X_test = X[6792:]
y_train = y[:6792]
y_test = y[6792:]
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30 )
from sklearn.model_selection import GridSearchCV

alphas = [0.000001,0.00001,0.0001,0.001,0.01,1,10,20,30,40,100]
enet = linear_model.ElasticNet(l1_ratio=0.2)
train_errors = list()
test_errors = list()
parameters = {'alpha':alphas}
estimator = GridSearchCV(enet,
                         parameters,cv=KFold(n_splits=5,shuffle=False))
estimator.fit(X, y)

import matplotlib.pyplot as plt


plt.figure()
plt.style.use('ggplot')
plt.xlabel("Alphas",fontsize=12)
plt.ylabel("score",fontsize=12)
plt.title("Elastic Net",fontsize=14)
plt.plot(alphas,estimator.cv_results_['mean_train_score'] ,'r')
plt.plot(alphas, estimator.cv_results_['mean_test_score'],'b')
red_patch = mpatches.Patch(color='red', label='Train Score')
blue_patch = mpatches.Patch(color='blue', label='validation Score')
plt.legend(handles=[red_patch, blue_patch])
plt.xscale('log')
plt.show()


plt.figure()
plt.style.use('seaborn')
plt.xlabel("Predicted Value",fontsize=12)
plt.ylabel("Residuals",fontsize=12)
plt.title("Elastic Net",fontsize=14)
pred_train=estimator.predict(X_train)
pred_test=estimator.predict(X_test)
plt.scatter(pred_train,pred_train-y_train,s=40,alpha=0.5 )
plt.hlines(y=0,xmin=0.6,xmax=1.1)
plt.scatter(pred_test,pred_test-y_test,s=40)
blue_patch = mpatches.Patch(color='blue', label='Train points')
green_patch = mpatches.Patch(color='green', label='Test points')
plt.legend(handles=[blue_patch, green_patch])
plt.show()


print ('Train score %.2f'%estimator.score(X_train, y_train) )
print ('Test score %.2f'%estimator.score(X_test, y_test) )

