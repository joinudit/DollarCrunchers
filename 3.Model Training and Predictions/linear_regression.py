
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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

#alpha = [1, .01, .02, .001, .0001, 10, 100]
# Initialize

train_error = []
val_error = []

X_train =  X_Norm[:7029]
X_test = X_Norm[7029:]
Y_train = Y[:7029]
Y_test = Y[7029:]


#X_train, X_test, Y_train, Y_test = train_test_split(X_Norm, Y, test_size=0.30, random_state=42)
model = linear_model.LinearRegression()
    #regr = model.fit(X_train, y_train)
params = {}
#scores = cross_validation.cross_val_score(regr, X_test, Y_test, cv=10)
grid = GridSearchCV(model, params, cv = 10 )
grid.fit(X_train, Y_train)

print "r2/variance :", grid.best_score_
print "root mean sq. error : %.2f" % np.sum((grid.predict(X_test)- Y_test)**2)

for x in grid.predict(X_test):
    print x



