
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,KFold
import matplotlib.patches as mpatches

def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

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
var = []
alpha = [1, 0.1, .01, .001, .0001, 10, 20, 30]
max_iteration = 100

X_train =  X_Norm[:7029]
X_test = X_Norm[7029:]    
Y_train = Y[:7029]
Y_test = Y[7029:]    
    
for x in alpha:
    train_error = []
    val_error = []
    
    
    X_train =  X_Norm[:6792]
    X_test = X_Norm[6792:]    
    Y_train = Y[:6792]
    Y_test = Y[6792:]    
    

 #   X_train, X_test, Y_train, Y_test = train_test_split(X_Norm, Y, test_size=0.30, random_state=42)
    regr = linear_model.RidgeCV(cv = KFold(n_splits = 5, shuffle = False), alphas = [x])
    regr.fit(X_train,Y_train)
    pred_train = pd.DataFrame(regr.predict(X_train)) 
    pred_test = pd.DataFrame(regr.predict(X_test))
    score=regr.score(X_test,Y_test)
    print "r2/variance :", regr.score(X_test,Y_test)
    var.append(score)    
    print "root mean sq. error : %.2f" % np.sum((regr.predict(X_test)- Y_test)**2)
r2 = np.sort(var)
t = np.argmax(r2)
print r2

for x in regr.predict(X_test):
    print x


