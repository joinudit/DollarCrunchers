import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import sqrt
from pylab import *

def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

# get data
top_ranked_features_features = []
data_path = 'Data'
df=readExcel(data_path + '/output_train.xlsx','Sheet1')

# remove correlated features
df = df.drop("energy",axis=1).drop("aud_us",axis=1) \
	.drop("urban_population",axis=1) \
	.drop("world_population",axis=1) \
	.drop("jpy_us",axis=1) \
	.drop("cny_us",axis=1) \
	.drop("eur_us",axis=1) \
	.drop("gold_price",axis=1) \
	.drop("export_us",axis=1).drop("import_us",axis=1)

# split data
Y= pd.DataFrame(df["cad_us"])
X=df.drop("date",axis=1).drop("cad_us",axis=1)
features = list(X.columns.values)

# normalize
min_max_scaler = preprocessing.MinMaxScaler()
X_Norm = min_max_scaler.fit_transform(X)
X_Norm = pd.DataFrame(X_Norm, columns = features)

# select the best features
kbest = SelectKBest(f_regression, k=14)
kbest.fit_transform(X_Norm, Y)

top_ranked_features = sorted(enumerate(kbest.scores_),key=lambda x:x[1], reverse=True)
sorted_features = []
sorted_scores = []

for i in range(0,14):
	index = top_ranked_features[i][0]
	sorted_features.append(features[index])
	sorted_scores.append(top_ranked_features[i][1])	


print sorted_features
print sorted_scores 

# plot feature scores
pos1 = range(len(sorted_features),len(sorted_features[:len(sorted_features) - 7]), -1) 
pos2 = range(len(sorted_features[:len(sorted_features) - 7]),0, -1)  
pos =  range(len(sorted_features),0, -1)
barh(pos1, sorted_scores[:7] , align='center', color='green')
barh(pos2, sorted_scores[7:] , align='center', color='red')
yticks(pos, sorted_features)
title("K Best")
xlabel('Feature Score')
show()


