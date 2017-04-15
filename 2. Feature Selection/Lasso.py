import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
import pandas as pd
from sklearn import preprocessing
import matplotlib.patches as mpatches
import numpy as np
from math import sqrt
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

train_error = []
test_error = []
train_variance = []
test_variance = []
alphas = [1, .01, .001,.0007, .0006, .00008, .00005, .000001]
feature_count = []
final_params = []
alphas = [1, .01, .001,.0007, .0006, .00008, .00005, .000001]

# perform Lasso
for alpha in alphas:
	regr = LassoCV(alphas = [alpha], cv= 5)
	regr.fit(X_train, Y_train)
	

	if alpha == .0005:		
		for i in range(0,len(regr.coef_)):
			if regr.coef_[i] != float(0):
				final_params.append(features[i])

	count = 0			
	for i in range(0,len(regr.coef_)):
			if regr.coef_[i] != float(0):
				count = count + 1

	feature_count.append(count)	
							
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

# print error and feature count
k_best = feature_count[test_error.index(min(test_error))]
print("Train squared error: " + ", ".join(str(x) for x in train_error))
print("Test squared error: " + ", ".join(str(x) for x in test_error))
print("Feature Count: " + ", ".join(str(x) for x in feature_count))
print("Feature Count: " + ", ".join(str(x) for x in feature_count))
print("Alphas: " + ", ".join(str(x) for x in alphas))
print "Best Training Error: " + str(min(train_error))
print "Best Test Error: " + str(min(test_error))
print "Best Alpha: " + str(alphas[test_error.index(min(test_error))])
print "Best Feature Count: " + str(k_best)

# plot figure
plt.figure()
plt.title("Feature Selection: Lasso", fontsize=14)
plt.xlabel("Number Of Features Selected",fontsize=12)
plt.ylabel("RMSE",fontsize=12)
plt.plot(feature_count, train_error, 'r')
plt.plot(feature_count, test_error, 'b')
plt.annotate('Optimal Count', xy=(k_best, min(test_error)), xytext=(k_best+2, min(test_error)+0.004),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
red_patch = mpatches.Patch(color='red', label='Train Error')
blue_patch = mpatches.Patch(color='blue', label='Validation Error')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

