# visualize correlation

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns;

# generate correlation matrix
def plot_correlation_map( df ):
    corr = df.corr()
    f, ax = plt.subplots( figsize =( 16 , 16 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .5 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 9 }
    )
    sns.plt.show()# -*- coding: utf-8 -*-


# read final data file
def readExcelSheet(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df


file_path='Data/output_final.xlsx'
df=readExcelSheet(file_path, 'Sheet1')
plot_correlation_map(df)

