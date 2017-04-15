import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from pylab import *

def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

data_path='Data'
df=readExcel(data_path+'/output_train.xlsx','Sheet1')

y=(df.cad_us).values
#remove Date column and Predicted Column (CAD-USD)
df=df.drop("date",axis=1).drop("cad_us",axis=1)
#remove Highly Correlated features
df=df.drop("energy",axis=1).drop("aud_us",axis=1).drop("urban_population",axis=1).drop("world_population",axis=1).drop("jpy_us",axis=1).drop("cny_us",axis=1).drop("eur_us",axis=1).drop("gold_price",axis=1).drop("export_us",axis=1).drop("import_us",axis=1)
X=df.values
#list of features
features=list(df.columns.values)
#Normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#Divide the data into train and test
X_train =  X[:5500]
X_test = X[5500:]
y_train = y[:5500]
y_test = y[5500:]

clf = linear_model.Ridge(alpha =30)
#clf=linear_model.LinearRegression()
rfecv = RFECV(estimator=clf, step=1, cv=KFold(5,shuffle=True)
             )

rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)
for i in np.array(features)[rfecv.support_]:
    print i 

pred_train = rfecv.predict(X_train) 
pred_test = rfecv.predict(X_test)

print ("Train score :%.2f" %rfecv.score(X_train,y_train))
print ("validation score :%.2f" %rfecv.score(X_test,y_test))

sorted_features=[]
sorted_scores=sorted(rfecv.ranking_)
for i in np.argsort(rfecv.ranking_):
    sorted_features.append(features[i])
sorted_scores=np.array(sorted_scores)
sorted_scores=6-sorted_scores
# plot feature scores
pos1 = range(len(sorted_features),len(sorted_features[:len(sorted_features) - 11]), -1) 
pos2 = range(len(sorted_features[:len(sorted_features) - 11]),0, -1)  
pos =  range(len(sorted_features),0, -1)
barh(pos1, sorted_scores[:11] , align='center', color='green')
barh(pos2, sorted_scores[11:] , align='center', color='red')
yticks(pos, sorted_features)
xlabel('Feature Rank')
title('RFE')
show()


