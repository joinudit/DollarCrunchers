# file to integrate records that have yearly values
# join records based on date

import pandas as pd
import numpy as np

# read entire sheet
def readExcelSheet(file,sheet):
    xl = pd.ExcelFile(file)
    df = xl.parse(sheet)
    return df

# read data for supplied columns
def readExcel(file, sheet, columns):
	xl = pd.ExcelFile(file)
	df = xl.parse(sheet)
	return df[columns]

data_path='../../Data'

# read deposit and lending rates
df_inr=readExcel(data_path + '/Deposit_LendingRates/interest_rates.xlsx', 'Sheet1', ['Year','Deposit Rate','Lending Rate'])
df_inr=df_inr.set_index('Year')

# read and combine GDP 
df_gdp=readExcel(data_path + '/GDP_Inflation_FDI/GDP.xlsx', 'Sheet1', ['Year','Amount'])
df_gdp=df_gdp.rename(columns={'Amount':'gdp'})
df_gdp=df_gdp.set_index('Year')
df_inr = df_inr.join(df_gdp, how='inner')

# read and combine imports and exports
df_import_export=readExcel(data_path + '/Imports_Exports/import_export.xlsx', 'Sheet1', ['Year','ExportUs','Exportall','ImportUs','Importall'])
df_import_export=df_import_export.set_index('Year')
df_inr = df_inr.join(df_import_export, how='inner')

# read and combine fdi
df_fdi=readExcel(data_path + '/GDP_Inflation_FDI/FDI.xlsx', 'Sheet1', ['Year','FDI'])
df_fdi=df_fdi.set_index('Year')
df_inr = df_inr.join(df_fdi, how='inner')

# read and combine population
df_population=readExcel(data_path + '/Population/Population Historical.xlsx', 'Sheet1', ['Year','Population','Migrants (net)','Urban Population','World Population'])
df_population=df_population.set_index('Year')
df_inr = df_inr.join(df_population, how='left')

# combine weekly and yearly records
df_year=readExcelSheet( '../Daily_Weely/output_weekly.xlsx', 'Sheet2')
df_year=df_year.rename(columns={'Date_year':'Year','Date1':'Date'})
df_year=df_year.set_index('Year')
df_inr = df_inr.join(df_year, how='inner')
df_final=readExcelSheet('../Daily_Weely/output_weekly.xlsx', 'Sheet1')
df_final=df_final.set_index('Date')
df_inr=df_inr.set_index('Date')
df_final = df_final.join(df_inr, how='left')

# interpolate missing values
df_final= df_final.interpolate()
df_final=df_final.rename(columns={  'Deposit Rate' : 'deposit_rate',
											'Lending Rate' : 'lending_rate',
											'ExportUs' : 'export_us',
											'Exportall' : 'export_all',
											'ImportUs' : 'import_us',	
											'Importall' : 'import_all',
											'FDI' : 'fdi',	
											'Population' : 'population',
											'Migrants (net)' : 'migrants', 	
											'Urban Population' : 'urban_population',
											'World Population' : 'world_population'})

# write output file
writer = pd.ExcelWriter('output_yearly.xlsx')
df_final.to_excel(writer,'Sheet1')
writer.save()