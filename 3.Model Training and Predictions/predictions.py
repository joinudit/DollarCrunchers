import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import datetime
from matplotlib.dates import date2num

def readExcel(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

data_path='Data'
df=readExcel(data_path + '/prediction_scores.xlsx','Sheet1')
date = df['date']
cad_us = df['cad_us']
sgd = df['sgd']
poly = df['polynomial_deg_2']
linear = df['linear']
ridge = df['ridge']
elastic_net = df["elastic_net"]
bayes = df["bayesian ridge"]

plt.title("SGD Predictions for December 2016")
#plt.ylim([0.72,0.78])
plt.xlabel("Date",fontsize=12)
plt.ylabel("CAD to US$",fontsize=12)
plt.plot(date,cad_us, 'g.-', linewidth=2)
plt.plot(date,sgd, 'r.-', linewidth=2)
plt.plot(date,poly, 'b--', linewidth=1)
plt.plot(date,linear, 'y--', linewidth=1)
plt.plot(date,ridge, 'm--', linewidth=2)
plt.plot(date,elastic_net, 'C1--', linewidth=1)
plt.plot(date,bayes, 'k--', linewidth=1)


green_patch = mpatches.Patch(color='green', label='Actual Values')
red_patch = mpatches.Patch(color='red', label='SGD')
blue_patch = mpatches.Patch(color='blue', label='Polynomial Regression - Degree 2')
yellow_patch = mpatches.Patch(color='yellow', label='Linear Regression')
magenta_patch = mpatches.Patch(color='magenta', label='Ridge Regression')
cyan_patch = mpatches.Patch(color='cyan', label='Elastic Net')
orange_patch = mpatches.Patch(color='orange', label='Lasso')
black_patch = mpatches.Patch(color='black', label='Bayes Regression')
plt.legend(handles=[green_patch, red_patch, blue_patch, yellow_patch, magenta_patch, cyan_patch, orange_patch, black_patch])
plt.show()
