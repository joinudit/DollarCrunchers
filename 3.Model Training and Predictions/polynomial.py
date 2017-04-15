from sklearn.linear_model import SGDRegressor
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import pickle
from sklearn.utils import shuffle
from math import sqrt


def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

def trainModel(X_train, Y_train):
	param_grid = [ {'alpha': [10**-8] ,
				'eta0': [.1], 
				'n_iter': [500]} ]  
	model = SGDRegressor(loss = 'squared_loss', penalty='l1', shuffle = True)
	clf = GridSearchCV(estimator = model, param_grid = param_grid, cv = 10)
	clf.fit(X_train, Y_train)
	return clf

# get data
data_path='Data'
df=readExcel(data_path + '/output_final.xlsx','Sheet1')

# select final features after feature selection
final_features = ['oil_price', 'metal', 'agriculture', 'import_all', 'migrants', 'gdp', 'population', 'export_all', 'lending_rate']
X = df[final_features]

# select label
label = "cad_us"
Y = df[label]

# polynomial regression
poly = preprocessing.PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

# normalize
min_max_scaler = preprocessing.MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)
X_norm = pd.DataFrame(X_norm)

# split training and test data. Training score 
split = 7029
X_train = X_norm[:split]
X_test = X_norm[split:]
Y_train = Y[:split]
Y_test = Y[split:]

# shuffle data
X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

# train model
clf = trainModel(X_train, Y_train) 
pickle.dump(clf, open("SGD_pol2.model", 'wb'))

# test
score_test = clf.score(X_test, Y_test) 
score_train = clf.score(X_train, Y_train) 
prediction = pd.DataFrame(clf.predict(X_test))

# print
print "Training Score: " + str(score_train)
print "Validation Score: " + str(clf.best_score_)
print "Test Score: " + str(score_test)
print "Test SSE: " + str(sqrt(np.mean((prediction.as_matrix() - Y_test.as_matrix()) ** 2)))
print prediction[0]