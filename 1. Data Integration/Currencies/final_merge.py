# final merge for currencies and inflation

import pandas as pd
import numpy as np

# method to read excel files
def readExcel(file, sheet, columns):
	xl = pd.ExcelFile(file)
	df = xl.parse(sheet)
	return df[columns]

#name of folder containing the xls
data_path = '../../Data'

# list of columns combined so far
lst_output_columns = ['Date','cad_us','gold_price','oil_price','energy','metal','forest','agriculture','fish','deposit_rate','lending_rate','gdp','export_us','export_all'	,'import_us',	'import_all','fdi','population','migrants','urban_population','world_population']
df_output=readExcel('../Yearly/output_yearly.xlsx', 'Sheet1', lst_output_columns)
df_output = df_output.rename(columns={'Date' : 'date'})
df_output=df_output.set_index('date')

# read australian dollar
df_aud= readExcel(data_path + '/ExchangeRates/AUD_USD.xlsx', 'Sheet1', ['date', 'aud_us'])
df_aud=df_aud.set_index('date')

# read chinese yen
df_cny= readExcel(data_path + '/ExchangeRates/CNY_USD.xlsx', 'Sheet1', ['date', 'cny_us'])
df_cny=df_cny.set_index('date')

# read euro
df_eur= readExcel(data_path + '/ExchangeRates/EUR_USD.xlsx', 'Sheet1', ['date', 'eur_us'])
df_eur=df_eur.set_index('date')

# read japanese yen
df_jpy= readExcel(data_path + '/ExchangeRates/JPY_USD.xlsx', 'Sheet1', ['date', 'jpy_us'])
df_jpy=df_jpy.set_index('date')

# read inflation
df_inflation= readExcel(data_path + '/GDP_Inflation_FDI/Inflation.xlsx', 'Sheet1', ['date', 'inflation'])
df_inflation=df_inflation.set_index('date')

# merge data
df_output = df_output.join(df_aud, how='left')
df_output = df_output.join(df_cny, how='left')
df_output = df_output.join(df_eur, how='left')
df_output = df_output.join(df_jpy, how='left')
df_output = df_output.join(df_inflation, how='left')

# interpolate data
df_output = df_output.interpolate()

# write final output file
writer = pd.ExcelWriter('output_final.xlsx')
df_output.to_excel(writer,'Sheet1')
writer.save()