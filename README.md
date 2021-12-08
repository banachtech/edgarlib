# Install the library
run `pip install -e git+https://github.com/banachtech/edgarlib#egg=Edgar`

# Functions
`update_csv()`

  return: csv files in folder 'EdgarData'
  
  You need to run this function for the first time you use this library so that the folder is downloaded in your local machine.

`get_company_details(ticker)`

  return: pandas dataframe or empty dictionary (if company is not found)

    The dataframe consists of the information of the company:
      1. Ticker
      2. Security
      3. Sector
      4. Sub-Industry     
      5. CIK   
      6. Cash / Total Asset
      7. ShortTermDebt / Total Asset
      8. LongTermDebt / Total Asset
      9. Wavg. of RDSGA / Total Assets
      10. β (Mkt-Rf) (F/F 3-Factor Model)
      11. β (Smb) (F/F 3-Factor Model)
      12. β (Hml) (F/F 3-Factor Model)
      13. Residuals
      14. β (Mom) (F/F Momentum Model)
      15. Score
  
  input:
      ticker (str): company's ticker