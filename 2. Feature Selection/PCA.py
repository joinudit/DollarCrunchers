import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from math import sqrt

# method to read excel file
def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

# get data
top_ranked_features_features = []
df=readExcel('Data/output_train.xlsx','Sheet1')

# remove correlated features
df = df.drop("energy",axis=1).drop("aud_us",axis=1) \
	 .drop("urban_population",axis=1).drop("world_population",axis=1) \
	 .drop("jpy_us",axis=1).drop("cny_us",axis=1) \
	 .drop("eur_us",axis=1).drop("gold_price",axis=1) \
	 .drop("export_us",axis=1).drop("import_us",axis=1)

# split data
Y= pd.DataFrame(df["cad_us"])
X=df.drop("date",axis=1).drop("cad_us",axis=1)
features = list(X.columns.values)

# normalize
min_max_scaler = preprocessing.MinMaxScaler()
X_Norm = min_max_scaler.fit_transform(X)
X_Norm = pd.DataFrame(X_Norm, columns = features)

# Initialize
list_k = range(3,15)
train_error = []
val_error = []

# PCA for different values of K
for f_count in list_k: 
	pca = PCA(n_components=f_count)

	# get features selected by PCA
	X_new = pd.DataFrame(pca.fit_transform(X))

	# split training and validation data
	X_train =  X_new[:5500]
	X_test = X_new[5500:]
	Y_train = Y[:5500]
	Y_test = Y[5500:]

	# train regression model
	regr = linear_model.RidgeCV(alphas=[1.8], cv =5)
	regr.fit(X_train,Y_train)
	pred_train = pd.DataFrame(regr.predict(X_train)) 
	pred_test = pd.DataFrame(regr.predict(X_test))

	# get training and test error
	tr_error = sqrt(np.mean((pred_train.as_matrix() - Y_train.as_matrix()) ** 2))
	te_error = sqrt(np.mean((pred_test.as_matrix() - Y_test.as_matrix()) ** 2))
	
	train_error.append(tr_error)
	val_error.append(te_error)

# print error and feature count
best_k = list_k[val_error.index(min(val_error))]
print("Train squared error: " + ", ".join(str(x) for x in train_error))
print("Validation squared error: " + ", ".join(str(x) for x in val_error))
print list_k 
print
print "Best Training Error: " + str(min(train_error))
print "Best Test Error: " + str(min(val_error))
print "Best K: " + str(best_k)

# plot figure
plt.figure()
plt.title("Dimensionality Reduction: PCA", fontsize=14)
plt.xlabel("Number of Dimensions", fontsize=12)
plt.ylabel("RMSE",fontsize=12)
plt.plot(list_k, train_error, 'r')
plt.plot(list_k, val_error, 'b')
plt.annotate('Optimal Count', xy=(best_k, min(val_error)), xytext=(best_k+1, min(val_error)-0.0005),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
red_patch = mpatches.Patch(color='red', label='Train Error')
blue_patch = mpatches.Patch(color='blue', label='Validation Error')
plt.legend(handles=[red_patch, blue_patch])
plt.show()