
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.feature_selection import RFE
import matplotlib.patches as mpatches
from math import sqrt

# read excel file
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
df=df.drop("energy",axis=1).drop("export_us",axis=1).drop("import_us",axis=1).drop("aud_us",axis=1).drop("urban_population",axis=1).drop("world_population",axis=1).drop("jpy_us",axis=1).drop("cny_us",axis=1).drop("eur_us",axis=1).drop("gold_price",axis=1)
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


train_error=[]
test_error=[]
best_test_error=float("inf")
best_features=[]
optimal_features=[]
for no_of_features in xrange(1,len(features)+1):  
    clf = linear_model.RidgeCV(alphas =[10,30,40,50,100,200,300],cv=5)
    
    rfecv = RFE(estimator=clf,n_features_to_select =no_of_features, step=1)

    rfecv.fit(X_train, y_train)
    pred_train = rfecv.predict(X_train) 
    pred_test = rfecv.predict(X_test)
    
    error_train=sqrt(np.mean((pred_train - y_train) ** 2)) 
    error_test=sqrt(np.mean((pred_test - y_test) ** 2)) 
    train_error.append(error_train)
    test_error.append(error_test)
    optimal_features.append(rfecv.n_features_)
    if best_test_error > error_test:
        best_test_error=error_test
        best_no_of_features=no_of_features
        best_features=np.array(features)[rfecv.support_]
    
         
# Plot number of features VS. cross validation error
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number Of Features Selected",fontsize=12)
plt.ylabel("RMSE",fontsize=12)
plt.title("Feature Selection: RFE",fontsize=14)
plt.plot(optimal_features, train_error,'r')
plt.plot(optimal_features, test_error,'b')
red_patch = mpatches.Patch(color='red', label='Train Error')
blue_patch = mpatches.Patch(color='blue', label='Validation Error')
plt.annotate('Optimal Count', xy=(best_no_of_features, best_test_error), xytext=(best_no_of_features+1, best_test_error+0.01),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.legend(handles=[red_patch, blue_patch])
plt.show()


print "Best features: "
for i in best_features:
    print i 
print "\nOptimal features:"
print best_no_of_features
print "\nMinimum test square mean error:"
print best_test_error
