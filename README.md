# DollarCrunchers
Major Factors Effecting CAS - US exchange rates

Public Github Repository: https://github.com/joinudit/DollarCrunchers

As the folder structure shows the Project Code has been divided into 5 parts:

1. Data: The information about Data Sources, Units and Metadata can be found under the file Metadata.docx present under this folder.

2. Data Integration: Date attribute is used to join various columns (daily, monthly and yearly intervals). To test the files both 
Data and Data Integration folders are needed to be present at the same folder level.

3. Feature Selection: 4 feature selection methods are used KBest, Lasso, PCA and RFE. Each method has its own python file and can be tested independetly by just running the code. A separate Data folder is present under feature selection which is needed to test the files.
Output and Visaulizations are generated at the end of each python file. Visualiztions are also saved in the folders.

4. Model Training and Predictions: 6 models (Ridge, Stochastic Gradient Descend, Lasso, Polynomial Regression, Elastic Net and Linear Regression) are trained. Each method has its own python file and can be tested independetly by just running the code. A separate Data folder is present under Model Training which is needed to test the files. 
Output and Visaulizations are generated at the end of each python file. Visualiztions are also saved in the folders.

5. Currency Time Series Plot: D3.js is used to visualize the correlation of CAD with other currencies. timeseries.html contains the html and d3 code. currency_date.tsv file contains the data for this visulization.  Visualiztions are also saved in the folders.
