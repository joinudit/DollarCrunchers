# file to integrate records that have daily values
# join records based on date

import pandas as pd
import numpy as np

# method to read excel files
def readExcel(file, sheet, columns):
	xl = pd.ExcelFile(file)
	df = xl.parse(sheet)
	return df[columns]

data_path = '../../Data'

# read cad_us currency rates
df_cad=readExcel(data_path + '/ExchangeRates/CAD_USD.xlsx', 'Sheet1', ['Date', 'cad_us'])
df_cad=df_cad.set_index('Date')

# read and integrate gold rates
df_gold= readExcel(data_path + '/Gold/Gold.xlsx', 'Sheet1', ['Date', 'gold_price'])
df_gold=df_gold.set_index('Date')
df_cad_gold = df_cad.join(df_gold, how='inner')

# read and merge oil rates
df_oil=readExcel(data_path + '/Oil/Oil.xlsx','Sheet1', ['Date', 'Price'])
df_oil=df_oil.set_index('Date')
df_oil=df_oil.rename(columns={'Price':'oil_price'})
df_cad_gold_oil=df_cad_gold.join(df_oil,how='inner')

# read and merge commodities
df_commodity=readExcel(data_path + '/Commodities/Commodities.xlsx','Sheet1', ['Date','W.ENER','W.MTLS','W.FOPR','W.AGRI','W.FISH'])
df_commodity=df_commodity.set_index('Date')
df_commodity=df_commodity.rename(columns={'Price':'oil_price', 
										    'W.ENER':'energy', 
										   	'W.MTLS':'metal', 
										   	'W.FOPR':'forest'
								, 			'W.AGRI':'agriculture', 
											'W.FISH':'fish'})
df_merged=df_cad_gold_oil.join(df_commodity,how='left')

# interpolate commodities as they are weekly records
df_merged= df_merged.interpolate()
df_merged['Date1']= df_merged.index
df_merged['Date_year']=map(lambda x:x.year,df_merged['Date1'])
df_temp= df_merged.loc[:,df_merged.columns.isin(['Date1','Date_year'])]
df_temp= df_temp.groupby('Date_year').min()

# write outputs
writer = pd.ExcelWriter('output_weekly.xlsx')
df_merged.to_excel(writer,'Sheet1')
df_temp.to_excel(writer,'Sheet2')
writer.save()