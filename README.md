# Installation
run `pip install -e git+https://github.com/banachtech/edgarlib#egg=Edgar`

# Functions
`update_csv()`

  return: csv files in folder 'EdgarData'
  
  You need to run this function for the first time you use this library so that the folder is downloaded in your local machine.

`get_company_details(ticker)`

  return: pandas dataframe or empty dictionary (if company is not found)

  * The dataframe consists of the information of the company:
    * Ticker
    * Security
    * Sector
    * Sub-Industry     
    * CIK   
    * Cash / Total Asset
    * ShortTermDebt / Total Asset
    * LongTermDebt / Total Asset
    * Wavg. of RDSGA / Total Assets
    * β (Mkt-Rf) (F/F 3-Factor Model)
    * β (Smb) (F/F 3-Factor Model)
    * β (Hml) (F/F 3-Factor Model)
    * Residuals
    * β (Mom) (F/F Momentum Model)
    * Score
  
  input:
  * ticker (str): company's ticker