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

def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

# get data
top_ranked_features_features = []
data_path='Data'
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
train_error = []
test_error = []
train_variance = []
test_variance = []
final_params = []

# select the best features
list_k = range(1,15,1)

for f_count in list_k:
	kbest = SelectKBest(f_regression, k=f_count)
	kbest.fit_transform(X_Norm, Y)
	top_ranked_features = sorted(enumerate(kbest.scores_),key=lambda x:x[1], reverse=True)[:f_count]
	top_ranked_features_features = []

	for i in range(0,f_count):
		index = top_ranked_features[i][0]
		top_ranked_features_features.append(features[index])	

	X_new = X_Norm[top_ranked_features_features]
	
	if f_count == 7:
		final_params = top_ranked_features_features

	# ridge regression
	X_train =  X_new[:5500]
	X_test = X_new[5500:]
	Y_train = Y[:5500]
	Y_test = Y[5500:]

	regr = linear_model.RidgeCV(alphas=[1.8], cv =5)
	regr.fit(X_train, Y_train)
	pred_train = pd.DataFrame(regr.predict(X_train)) 
	pred_test = pd.DataFrame(regr.predict(X_test))

	tr_error = sqrt(np.mean((pred_train.as_matrix() - Y_train.as_matrix()) ** 2))
	tr_var = regr.score(X_train, Y_train)
	te_error = sqrt(np.mean((pred_test.as_matrix() - Y_test.as_matrix()) ** 2))
	te_var = regr.score(X_test, Y_test)

	train_error.append(tr_error)
	train_variance.append(tr_var)
	test_error.append(te_error)
	test_variance.append(te_var)

# print error and best features
best_k = list_k[test_error.index(min(test_error))]
print("Train squared error: " + ", ".join(str(x) for x in train_error))
print("Test squared error: " + ", ".join(str(x) for x in test_error))
print("K: " + ", ".join(str(x) for x in list_k))
print
print "Best Training Error: " + str(min(train_error))
print "Best Test Error: " + str(min(test_error))
print "Best K: " + str(best_k)
print ("Best Parameters: " + ", ".join(str(x) for x in final_params))

# plot graph
plt.figure()
plt.title("Feature Selection: K Best", fontsize=14)
plt.xlabel("Number Of Features Selected", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.plot(list_k, train_error, 'r')
plt.plot(list_k, test_error, 'b')
red_patch = mpatches.Patch(color='red', label='Train Error')
blue_patch = mpatches.Patch(color='blue', label='Validation Error')
plt.annotate('Optimal Count', xy=(best_k, min(test_error)), xytext=(best_k+1, min(test_error)+0.0006),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.legend(handles=[red_patch, blue_patch])
plt.show()