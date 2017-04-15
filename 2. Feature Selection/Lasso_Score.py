import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
import pandas as pd
from sklearn import preprocessing
import matplotlib.patches as mpatches
import numpy as np
from math import sqrt
from pylab import *

def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

# get data
top_ranked_features_features = []
df=readExcel('Data/output_train.xlsx','Sheet1')

# remove correlated features
df = df.drop("energy",axis=1) \
	.drop("aud_us",axis=1).drop("urban_population",axis=1) \
	.drop("world_population",axis=1).drop("jpy_us",axis=1) \
	.drop("cny_us",axis=1).drop("eur_us",axis=1).drop("gold_price",axis=1) \
	.drop("export_us",axis=1).drop("import_us",axis=1)

# split data
Y= pd.DataFrame(df["cad_us"])
X=df.drop("date",axis=1).drop("cad_us",axis=1)
features = list(X.columns.values)

# normalize
min_max_scaler = preprocessing.MinMaxScaler()
X_Norm = min_max_scaler.fit_transform(X)
X_Norm = pd.DataFrame(X_Norm, columns = features)

X_train =  X_Norm[:5500]
X_test = X_Norm[5500:]
Y_train = Y[:5500]
Y_test = Y[5500:]

print "###############LASO#####################"
train_error = []
test_error = []
alphas = [.000001]
feature_count = []
final_params = []
final_coeff = []

# perform lasso 
for alpha in alphas:
	regr = LassoCV(alphas = [alpha], cv= 5)
	regr.fit(X_train, Y_train)

	if alpha == .0006:		
		for i in range(0,len(regr.coef_)):
			if regr.coef_[i] != float(0):
				final_params.append(features[i])

		final_coeff = regr.coef_

	count = 0			
	for i in range(0,len(regr.coef_)):
			if regr.coef_[i] != float(0):
				count = count + 1

	feature_count.append(count)	
							
	pred_train = pd.DataFrame(regr.predict(X_train)) 
	pred_test = pd.DataFrame(regr.predict(X_test))

	tr_error = sqrt(np.mean((pred_train.as_matrix() - Y_train.as_matrix()) ** 2))
	te_error = sqrt(np.mean((pred_test.as_matrix() - Y_test.as_matrix()) ** 2))

	train_error.append(tr_error)
	test_error.append(te_error)

k_best = feature_count[test_error.index(min(test_error))]

scores = map(abs, regr.coef_)
sorted_scores = sorted(map(abs, regr.coef_), reverse=True)
features_enum = sorted(enumerate(scores),key=lambda x:x[1], reverse=True) 

sorted_features = []
for i in range(0,14):
	index = features_enum[i][0]
	sorted_features.append(features[index])	

# plot feature scores
pos1 = range(len(sorted_features),len(sorted_features[:len(sorted_features) - 8]), -1) 
pos2 = range(len(sorted_features[:len(sorted_features) - 8]),0, -1)  
pos =  range(len(sorted_features),0, -1)
barh(pos1, sorted_scores[:8] , align='center', color='green')
barh(pos2, sorted_scores[8:] , align='center', color='red')
yticks(pos, sorted_features)
title("Lasso")
xlabel('Feature Score')
show()