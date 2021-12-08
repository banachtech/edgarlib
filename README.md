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

`get_overall_rank(value=20)`

  return: pandas dataframe
  * The dataframe consists of the ranking of the companies in S&P 500 based on score.
  
  input:
  * value (int): number of rows wanted to show. Default: 20.

`get_additional_analytics(tickers)`

  return: pandas dataframe or empty dictionary (if company is not found)
  * The dataframe consists of the information:
    * 12-month momentum
    * 3-month momentum
    * 30-day annualized volatility
    * Maximum return
    * Minimum return
  
  input:
  * tickers (str or list): single company's ticker or list of companies' ticker.

`get_quarter_details(symbol)`

  return: pandas dataframe or empty dictionary (if company is not found)
  * The dataframe consists of the information of the company:
    * Date
    * Cash / Total Asset
    * ShortTermDebt / Total Asset
    * LongTermDebt / Total Asset
    * β (Mkt-Rf) (F/F 3-Factor Model)
    * β (Smb) (F/F 3-Factor Model)
    * β (Hml) (F/F 3-Factor Model)
    * Residuals
    * β (Mom) (F/F Momentum Model)
  
  input:
  * ticker (str): company's ticker

`get_sector_comparison(ticker)`

  return: pandas dataframe or empty dictionary (if company is not found)
  * The dataframe consists of summary statistics of the corresponding GICS Sector:
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

`get_report_links(ticker)`

  return: dictionary
  * The dictionary consists of the link of statements of the company for the latest quarter.
  
  input:
  * ticker (str): company's ticker


`get_sorted_sector_comparison(ticker, variable="Score")`

  return: pandas dataframe or empty dictionary (if company is not found)
  * The dataframe consists of top 10's ranking based on the variable:
  
  input:
  * ticker (str): company's ticker
  * variable (str): 'Cash', 'RDSGA', 'Score', 'ShortTermDebt', or 'LongTermDebt'. Default: 'Score'