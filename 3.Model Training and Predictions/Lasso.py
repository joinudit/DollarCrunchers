import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
import pandas as pd
from sklearn import preprocessing
import matplotlib.patches as mpatches

def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

# get data
top_ranked_features_features = []
data_path='Data'
df=readExcel(data_path + '/output_final.xlsx','Sheet1')

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
train_variance = []
test_variance = []	
alphas = [1, .1, .01, .001, .0001, .00001]
final_coeff = []

for alpha in alphas:
	regr = LassoCV(alphas = [alpha], cv= 5)
	regr.fit(X_train, Y_train)

	if alpha == .001:		
		for i in range(0,len(regr.coef_)):
			if regr.coef_[i] != float(0):
				final_coeff.append(features[i])

	pred_train = pd.DataFrame(regr.predict(X_train)) 
	pred_test = pd.DataFrame(regr.predict(X_test))

	tr_error = sum((pred_train.as_matrix() - Y_train.as_matrix()) ** 2)
	tr_var = regr.score(X_train, Y_train)
	te_error = sum((pred_test.as_matrix() - Y_test.as_matrix()) ** 2)
	te_var = regr.score(X_test, Y_test)

	train_error.append(tr_error[0])
	train_variance.append(tr_var)
	test_error.append(te_error[0])
	test_variance.append(te_var)

print("Train squared error: " + ", ".join(str(x) for x in train_error))
print("Test squared error: " + ", ".join(str(x) for x in test_error))
print("Train Variance score: " + ", ".join(str(x) for x in train_variance)) # should be close to 1
print("Test Variance score: " + ", ".join(str(x) for x in test_variance)) # should be close to 1
print("Alphas: " + ", ".join(str(x) for x in alphas))
print final_coeff
print
print "Best Training Error: " + str(min(train_error))
print "Best Test Error: " + str(min(test_error))
print "Best Training Variance: " + str(max(train_variance))
print "Best Test Variance: " + str(max(test_variance))
print "Best Alpha: " + str(alphas[test_error.index(min(test_error))])

plt.figure()
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.gca().invert_xaxis()
plt.gca().set_xscale('symlog')
plt.semilogx(alphas, train_error, 'r')
plt.semilogx(alphas, test_error, 'b')
red_patch = mpatches.Patch(color='red', label='Train Error')
blue_patch = mpatches.Patch(color='blue', label='Test Error')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

print
print "###############RIDGE#####################"

X_train2 = X_train[final_coeff]
X_test2 = X_test[final_coeff]

alphas2 = [100, 90, 80, 70, 60, 50, 30, 20, 15, 10, 1, 0.1]
train_error2 = []
test_error2 = []
train_variance2 = []
test_variance2 = []

for alpha in alphas2:
	regr2 = RidgeCV(alphas=[alpha], cv=5)
	regr2.fit(X_train2, Y_train)
	pred_train2 = pd.DataFrame(regr2.predict(X_train2)) 
	pred_test2 = pd.DataFrame(regr2.predict(X_test2))

	tr_error2 = sum((pred_train2.as_matrix() - Y_train.as_matrix()) ** 2)
	tr_var2 = regr2.score(X_train2, Y_train)
	te_error2 = sum((pred_test2.as_matrix() - Y_test.as_matrix()) ** 2)
	te_var2 = regr2.score(X_test2, Y_test)

	train_error2.append(tr_error2[0])
	train_variance2.append(tr_var2)
	test_error2.append(te_error2[0])
	test_variance2.append(te_var2)

print("Train squared error: " + ", ".join(str(x) for x in train_error2))
print("Test squared error: " + ", ".join(str(x) for x in test_error2))
print("Train Variance score: " + ", ".join(str(x) for x in train_variance2)) # should be close to 1
print("Test Variance score: " + ", ".join(str(x) for x in test_variance2)) # should be close to 1
print alphas2
print
print "Best Training Error: " + str(min(train_error2))
print "Best Test Error: " + str(min(test_error2))
print "Best Training Variance: " + str(max(train_variance2))
print "Best Test Variance: " + str(max(test_variance2))
print "Best Alpha: " + str(alphas2[test_error2.index(min(test_error2))])

plt.figure()
plt.xlabel("Alpha")
plt.ylabel("Error")
plt.gca().invert_xaxis()
#plt.plot(alphas2, train_error2, 'r')
#plt.plot(alphas2, train_variance2, 'b')
plt.plot(alphas2, test_error2, 'r')
plt.plot(alphas2, test_variance2, 'b')
red_patch = mpatches.Patch(color='red', label='Error')
blue_patch = mpatches.Patch(color='blue', label='Variance')
plt.legend(handles=[red_patch, blue_patch])
plt.legend(handles=[red_patch, blue_patch])
plt.show()