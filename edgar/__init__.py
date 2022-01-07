import pandas as pd
import pandas_datareader.data as reader
from datetime import date
from yahooquery import Ticker
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import time
import os
import requests
import bs4 as bs

def update_csv():
    """
    return: csv files in folder 'EdgarData'
    
    You need to run this function for the first time you use this library so that the folder is downloaded in your local machine.
    """
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies') # get the table from wikipedia
        snp500_table = tables[0]
        tickers_list = snp500_table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'CIK']].values.tolist() # extract the data from table
        for i in range(len(tickers_list)):
            tickers_list[i][0] = tickers_list[i][0].replace('.', '-') # change . to - : since tickers in edgar database is using -

        datas = []
        
        # read the FF datas for the past five years => 3 factor model and momentum model
        f_f_three_factor_model = reader.DataReader('F-F_Research_Data_Factors', 
                                                'famafrench', 
                                                end=f"{date.today().year}-{date.today().month}-01", 
                                                start=f"{date.today().year - 5}-{date.today().month - 2}-01")

        f_f_momentum_model = reader.DataReader('F-F_Momentum_Factor', 
                                                'famafrench', 
                                                end=f"{date.today().year}-{date.today().month}-01", 
                                                start=f"{date.today().year - 5}-{date.today().month - 2}-01")

        # change the datetime to the same timestamp
        f_f_three_factor_model = f_f_three_factor_model[0]
        f_f_three_factor_model.index = f_f_three_factor_model.index.to_timestamp()

        f_f_momentum_model = f_f_momentum_model[0]
        f_f_momentum_model.index = f_f_momentum_model.index.to_timestamp()

        # concatenate the dataframe together
        f_f_models = pd.concat([f_f_three_factor_model, f_f_momentum_model], axis=1)
        f_f_models.columns = [i.replace(' ', '') for i in f_f_models.columns]

        for ticker in tickers_list:
            value = Ticker(ticker[0]) # get the info of company using yahooquery
            element = []
            element.append(ticker)
            element = [item for sublist in element for item in sublist] # [ticker, security, gics sector, sub-industry, cik]
            financials_annual = value.income_statement(frequency='a', trailing=False) # get the income statement, without ttm
            financials_annual = financials_annual.sort_values(by=['asOfDate'], ascending=False).T
            financials_annual.columns = [f"{financials_annual.columns[i]}_{i}" for i in range(len(financials_annual.columns))] # change the column name since all are the same
            balance_sheet_annual = value.balance_sheet(frequency='a', trailing=False).T.iloc[:, -1] # get the latest annual balance sheet
            balance_sheet_quarter = value.balance_sheet(frequency='q', trailing=False).T.iloc[:, -1] # get the latest quarter balance sheet
            
            weight_avg_RDSGA_annual = 0
            coefficient = [1, 0.75, 0.5, 0.25]
            for i in range(len(coefficient)):
                # get the variable 'ResearchAndDevelopment', check whether it is 0 or none
                try:
                    if financials_annual.loc['ResearchAndDevelopment', financials_annual.columns[i]] == None or np.isnan(financials_annual.loc['ResearchAndDevelopment', financials_annual.columns[i]]):
                        r_and_d_annual = 0 
                    else:
                        r_and_d_annual = financials_annual.loc['ResearchAndDevelopment', financials_annual.columns[i]]
                except:
                    r_and_d_annual = 0

                # get the variable 'SellingGeneralAndAdministration', check whether it is 0 or none
                try:
                    if financials_annual.loc['SellingGeneralAndAdministration', financials_annual.columns[i]] == None or np.isnan(financials_annual.loc['SellingGeneralAndAdministration', financials_annual.columns[i]]):
                        marketing_expenses_annual = 0 
                    else:
                        marketing_expenses_annual = financials_annual.loc['SellingGeneralAndAdministration', financials_annual.columns[i]]
                except:
                    marketing_expenses_annual = 0
                numerator = (r_and_d_annual + marketing_expenses_annual / 3) * coefficient[i]
                weight_avg_RDSGA_annual += numerator

            # get the variable 'TotalAssets', check whether it is 0 or none
            try:
                if balance_sheet_annual['TotalAssets'] == None or np.isnan(balance_sheet_annual['TotalAssets']):
                    total_assets_annual = 0 
                else:
                    total_assets_annual = balance_sheet_annual['TotalAssets']
            except:
                total_assets_annual = 0

            # get the variable 'TotalAssets', check whether it is 0 or none
            try:
                if balance_sheet_quarter['TotalAssets'] == None or np.isnan(balance_sheet_quarter['TotalAssets']):
                    total_assets_quarter = 0 
                else:
                    total_assets_quarter = balance_sheet_quarter['TotalAssets']
            except:
                total_assets_quarter = 0

            # get the variable 'CashAndCashEquivalents', check whether it is 0 or none
            try:
                if balance_sheet_quarter['CashAndCashEquivalents'] == None or np.isnan(balance_sheet_quarter['CashAndCashEquivalents']):
                    cash_quarter = 0 
                else:
                    cash_quarter = balance_sheet_quarter['CashAndCashEquivalents']
            except:
                cash_quarter = 0

            # get the variable 'CurrentDebt', check whether it is 0 or none
            try:
                if balance_sheet_quarter['CurrentDebt'] == None or np.isnan(balance_sheet_quarter['CurrentDebt']):
                    short_term_debt_quarter = 0 
                else:
                    short_term_debt_quarter = balance_sheet_quarter['CurrentDebt']
            except:
                short_term_debt_quarter = 0

            # get the variable 'LongTermDebt', check whether it is 0 or none
            try:
                if balance_sheet_quarter['LongTermDebt'] == None or np.isnan(balance_sheet_quarter['LongTermDebt']):
                    long_term_debt_quarter = 0 
                else:
                    long_term_debt_quarter = balance_sheet_quarter['LongTermDebt']
            except:
                long_term_debt_quarter = 0
            
            # calculate the variable 'cash_per_assets', if failed then return 0
            try:
                cash_per_assets = round(cash_quarter / total_assets_quarter * 100, 3)
            except:
                cash_per_assets = np.nan
            element.append(cash_per_assets)
            
            # calculate the variable 'st_debt_per_assets', if failed then return 0
            try:
                st_debt_per_assets = round(short_term_debt_quarter / total_assets_quarter * 100, 3)
            except:
                st_debt_per_assets = np.nan
            element.append(st_debt_per_assets)
            
            # calculate the variable 'lt_debt_per_assets', if failed then return 0
            try:
                lt_debt_per_assets = round(long_term_debt_quarter / total_assets_quarter * 100, 3)
            except:
                lt_debt_per_assets = np.nan
            element.append(lt_debt_per_assets)
            
            # calculate the variable 'weighted_avg_RDSGA_per_assets', if failed then return 0
            try:
                weighted_avg_RDSGA_per_assets = round(weight_avg_RDSGA_annual / total_assets_annual * 100, 3)
            except:
                weighted_avg_RDSGA_per_assets = np.nan
            element.append(weighted_avg_RDSGA_per_assets)
            
            value_hist = yf.Ticker(ticker[0]) # get the price history using yfinance
            hist = value_hist.history(interval="1mo", 
                                    end=f"{date.today().year}-{date.today().month}-01", 
                                    start=f"{date.today().year - 5}-{date.today().month - 2}-01") # get the price history dataframe
            hist = hist.dropna()
            hist = hist[['Close']] # only closing price is needed
            hist['Return'] = hist['Close'].rolling(window=2).apply(lambda x: x[1]/x[0] - 1) # calculating the return
            hist = hist.dropna()
            
            result = pd.concat([hist, f_f_models], axis=1).dropna()

            result['Return - RF'] = result.Return - result.RF
            
            # linear regression to calculate betas and residual of ff 3 factor model
            X_ff = result[['Mkt-RF', 'SMB', 'HML']].to_numpy()
            y_ff = result['Return - RF'].to_numpy()
            
            reg_ff = LinearRegression().fit(X_ff, y_ff)
            slope_MKT = round(reg_ff.coef_[0] * 100, 3)
            element.append(slope_MKT)
            slope_SMB = round(reg_ff.coef_[1] * 100, 3)
            element.append(slope_SMB)
            slope_HML = round(reg_ff.coef_[2] * 100, 3)
            element.append(slope_HML)
            result['FF Three Factors Model Return'] = reg_ff.predict(X_ff)
            result['Squared Error'] = (result['Return'] - result['FF Three Factors Model Return']) ** 2
            try:
                residuals = round(result['Squared Error'].mean() * 100, 3)
            except:
                residuals = np.nan
            element.append(residuals)
            
            # linear regression to calculate beta of ff momentum model
            X_momentum = result[['Mom']].to_numpy()
            y_momentum = result['Return'].to_numpy()

            reg_momentum = LinearRegression().fit(X_momentum, y_momentum)
            slope_MOM = round(reg_momentum.coef_[0] * 100, 3)
            element.append(slope_MOM)
            
            # calculate the overall score
            score = round((0.16 * cash_per_assets 
                        - 0.22 * st_debt_per_assets 
                        - 0.19 * lt_debt_per_assets 
                        + 0.15 * weighted_avg_RDSGA_per_assets 
                        + 0.17 * slope_MKT 
                        + 0.03 * slope_SMB 
                        + 0.02 * slope_HML 
                        - 0.5 * residuals 
                        - 0.05 * slope_MOM), 2)
            element.append(score)
            print(element)
            datas.append(element)
            
            del value
            del value_hist
            del financials_annual
            del balance_sheet_annual
            del balance_sheet_quarter
            del weight_avg_RDSGA_annual
            del r_and_d_annual
            del marketing_expenses_annual 
            del numerator
            del total_assets_annual
            del total_assets_quarter
            del cash_quarter
            del short_term_debt_quarter
            del long_term_debt_quarter
            del cash_per_assets
            del st_debt_per_assets
            del lt_debt_per_assets
            del weighted_avg_RDSGA_per_assets
            del hist
            del result
            del X_ff
            del y_ff
            del reg_ff
            del residuals
            del X_momentum
            del y_momentum
            del reg_momentum
            del slope_MKT
            del slope_SMB
            del slope_HML
            del slope_MOM
            del element
            time.sleep(3) # 3s delay for each loop

        try:
            os.makedirs('EdgarData') # create directory
        except:
            pass
        
        headers = ['Ticker', 
                'Security', 
                'GICS Sector', 
                'GICS Sub-Industry', 
                'CIK',
                'Cash / Total Assets (MRQ) (%)', 
                'Short-term Debt / Total Assets (MRQ) (%)', 
                'Long-term Debt / Total Assets (MRQ) (%)',
                'Weighted average of RDSGA / Total assets (MRY) (%)', 
                'Beta (Mkt-RF) (F-F 3 Factors Model) (%)', 
                'Beta (SMB) (F-F 3 Factors Model) (%)', 
                'Beta (HML) (F-F 3 Factors Model) (%)',
                'Residuals (F-F 3 Factors Model) (%)', 
                'Beta (MOM) (F-F Momentum Model) (%)', 
                'Score (%)']

        dataframe = pd.DataFrame(data=datas, columns=headers) # create dataframe to store the data
        dataframe = (dataframe
                    .sort_values(by=['GICS Sector', 'Ticker', 'GICS Sub-Industry'], ascending=True, na_position='first')
                    .reset_index(drop=True))
        
        dataframe = dataframe.dropna()
        dataframe.to_csv('EdgarData/S&P500.csv') # export the dataframe to csv file
        
        sector_list = dataframe['GICS Sector'].unique().tolist() # get the GICS sectors
        
        sectors = {}

        for i in range(len(sector_list)):
            sectors[i] = {}
            sectors[i]['sector'] = sector_list[i]
            sectors[i]['data'] = (dataframe[dataframe['GICS Sector'] == sector_list[i]]
                                    .sort_values(by=['GICS Sub-Industry', 'Ticker'], ascending=True, na_position='first')
                                    .reset_index(drop=True))
            sectors[i]['statistics'] = (sectors[i]['data']
                                        .describe()
                                        .T[['mean', 'std', '50%', 'min', 'max']]
                                        .round(3)
                                        .to_dict('index'))
            sectors[i]['data'].to_csv(f"EdgarData/S&P500_{sectors[i]['sector'].replace(' ', '_')}.csv") # export the sector data to csv files
        
        return 1
    except:
        return 0

def get_company_details(ticker):
    """
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
    """
    ticker = ticker.upper()
    headers = ['Ticker', 'Security', 'Sector', 'Sub-Industry', 'CIK', 
                'Cash', 'ShortTermDebt', 'LongTermDebt', 'RDSGA', 'Mkt', 'Smb', 'Hml', 'Residuals', 'Mom', 'Score']
    dataframe = pd.read_csv('EdgarData/S&P500.csv', index_col=0)
    dataframe.columns = headers # change the column's name to header

    data_wanted = dataframe[dataframe['Ticker'] == ticker.strip()].reset_index(drop=True) # get the row with ticker input

    if data_wanted.empty: # if not found
        return {}
   
    return data_wanted

def get_overall_rank(value=20):
    """
    return: pandas dataframe
        The dataframe consists of the ranking of the companies in S&P 500 based on score.
    
    input:
        value (int): number of rows wanted to show. Default: 20.
    """
    headers = ['Ticker', 'Security', 'Sector', 'Sub-Industry', 'CIK', 
                'Cash', 'ShortTermDebt', 'LongTermDebt', 'RDSGA', 'Mkt', 'Smb', 'Hml', 'Residuals', 'Mom', 'Score']
    dataframe = pd.read_csv(f"EdgarData/S&P500.csv", index_col=0)
    dataframe.columns = headers # change the column name to the header
    if int(value) >= len(dataframe): # if the value is greater than the maximum row
        dataframe = dataframe.sort_values(by=['Score', 'Ticker'], ascending=False).reset_index(drop=True)[['Ticker', 'Security', 'Sector', 'Score']]
    else:
        dataframe = dataframe.sort_values(by=['Score', 'Ticker'], ascending=False).reset_index(drop=True)[['Ticker', 'Security', 'Sector', 'Score']][:int(value)]
    result = dataframe
    return result

def get_additional_analytics(tickers):
    """
    return: pandas dataframe or empty dictionary (if company is not found)
        The dataframe consists of the information:
            1. 12-month momentum
            2. 3-month momentum
            3. 30-day annualized volatility
            4. Maximum return
            5. Minimum return
    
    input:
        tickers (str or list): single company's ticker or list of companies' ticker.
    """
    if isinstance(tickers, str): # if the input is a string (single company)
        tickers = [tickers.strip()]

    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    snp500_table = tables[0]
    tickers_list = snp500_table['Symbol'].to_list()
    for i in range(len(tickers_list)):
        tickers_list[i] = tickers_list[i].replace('.', '-') # get the ticker list

    for i in range(len(tickers)):
        tickers[i] = tickers[i].upper()
        if tickers[i] not in tickers_list:
            return {}

    data = {}

    for i in tickers:
        i = i.strip() # clear the spaces
        data[i] = {}
        ticker_hist = yf.Ticker(i) # get the price history using yfinance
        
        hist_1y = ticker_hist.history(period='1y') # 1 year data
        hist_1y = hist_1y.dropna()
        hist_1y = hist_1y['Close']

        hist_3mo = ticker_hist.history(period='3mo') # 3 month data
        hist_3mo = hist_3mo.dropna()
        hist_3mo = hist_3mo['Close']

        hist_30d = ticker_hist.history(period='30d') # 30 day data
        hist_30d = hist_30d.dropna()
        hist_30d = hist_30d[['Close']]
        hist_30d['Return'] = hist_30d['Close'].rolling(window=2).apply(lambda x: np.log(x[1]/x[0]))
        hist_30d = hist_30d.dropna()
        hist_30d = hist_30d['Return']
        
        mom12 = round((hist_1y[-1] - hist_1y[0]) / hist_1y[0], 3)
        mom3 = round((hist_3mo[-1] - hist_3mo[0]) / hist_3mo[0], 3)
        vol30 = round(np.std(hist_30d) * np.sqrt(252), 3)
        maxret = round(max(hist_30d), 3)
        minret = round(min(hist_30d), 3)
        
        data[i]['mom12'] = mom12
        data[i]['mom3'] = mom3
        data[i]['vol30'] = vol30
        data[i]['maxret'] = maxret
        data[i]['minret'] = minret
    
    data = pd.DataFrame.from_dict(data, orient='index', columns=['mom12', 'mom3', 'vol30', 'maxret', 'minret'])
    return data

def get_quarter_details(symbol):
    """
    return: pandas dataframe or empty dictionary (if company is not found)
        The dataframe consists of the information of the company:
            1. Date
            2. Cash / Total Asset
            3. ShortTermDebt / Total Asset
            4. LongTermDebt / Total Asset
            5. β (Mkt-Rf) (F/F 3-Factor Model)
            6. β (Smb) (F/F 3-Factor Model)
            7. β (Hml) (F/F 3-Factor Model)
            8. Residuals
            9. β (Mom) (F/F Momentum Model)
    
    input:
        ticker (str): company's ticker
    """
    symbol = symbol.strip() # clear the spaces
    symbol = symbol.upper() # change to uppercase
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    snp500_table = tables[0]
    tickers_list = snp500_table['Symbol'].to_list()
    for i in range(len(tickers_list)):
        tickers_list[i] = tickers_list[i].replace('.', '-')

    if symbol not in tickers_list: # if ticker is not in the ticker list, return empty
        return {}
    
    ticker = Ticker(symbol)
    ticker_hist = yf.Ticker(symbol)
    
    ticker_balance_sheet_quarter = ticker.balance_sheet(frequency='q', trailing=False).T
    ticker_balance_sheet_quarter.columns = [f"{ticker_balance_sheet_quarter.columns[i]}_{i}" for i in range(len(ticker_balance_sheet_quarter.columns))]
    ticker_balance_sheet_quarter = ticker_balance_sheet_quarter[[ticker_balance_sheet_quarter.columns[i] for i in range(len(ticker_balance_sheet_quarter.columns)-1, -1, -1)]]
    ticker_balance_sheet_quarter.columns = [f"{ticker_balance_sheet_quarter.columns[i]}" for i in range(len(ticker_balance_sheet_quarter.columns)-1, -1, -1)]
    
    data = {}
    for i in range(len(ticker_balance_sheet_quarter.columns)):
        data[f'Q({-i})'] = {}
        date = ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['asOfDate']
        data[f'Q({-i})']['date'] = (date.year, date.month, date.day)
        try:
            if ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['TotalAssets'] == None or np.isnan(ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['TotalAssets']):
                total_assets_quarter = 0
            else:
                total_assets_quarter = ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['TotalAssets']
        except:
            total_assets_quarter = 0

        try:
            if ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['CashAndCashEquivalents'] == None or np.isnan(ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['CashAndCashEquivalents']):
                cash_quarter = 0
            else:
                cash_quarter = ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['CashAndCashEquivalents']
        except:
            cash_quarter = 0

        try:
            if ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['CurrentDebt'] == None or np.isnan(ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['CurrentDebt']):
                short_term_debt_quarter = 0
            else:
                short_term_debt_quarter = ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['CurrentDebt']
        except:
            short_term_debt_quarter = 0

        try:
            if ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['LongTermDebt'] == None or np.isnan(ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['LongTermDebt']):
                long_term_debt_quarter = 0
            else:
                long_term_debt_quarter = ticker_balance_sheet_quarter[ticker_balance_sheet_quarter.columns[i]]['LongTermDebt']
        except:
            long_term_debt_quarter = 0
        
        if i == 0:
            hist = ticker_hist.history(interval="1mo", 
                                    end=f"{date.today().year}-{date.today().month}-01", 
                                    start=f"{date.today().year - 5}-{date.today().month - 2}-01")
            hist = hist.dropna()
            hist = hist[['Close']]
            hist['Return'] = hist['Close'].rolling(window=2).apply(lambda x: x[1]/x[0] - 1)
            hist = hist.dropna()

            f_f_three_factor_model = reader.DataReader('F-F_Research_Data_Factors', 
                                                    'famafrench', 
                                                    end=f"{date.today().year}-{date.today().month}-01", 
                                                    start=f"{date.today().year - 5}-{date.today().month - 2}-01")

            f_f_momentum_model = reader.DataReader('F-F_Momentum_Factor', 
                                                    'famafrench', 
                                                    end=f"{date.today().year}-{date.today().month}-01", 
                                                    start=f"{date.today().year - 5}-{date.today().month - 2}-01")
        else:
            hist = ticker_hist.history(interval="1mo", 
                                    end=f"{data[f'Q({-i})']['date'][0]}-{data[f'Q({-i})']['date'][1]}-01", 
                                    start=f"{data[f'Q({-i})']['date'][0] - 5}-{data[f'Q({-i})']['date'][1] - 2}-01")
            hist = hist.dropna()
            hist = hist[['Close']]
            hist['Return'] = hist['Close'].rolling(window=2).apply(lambda x: x[1]/x[0] - 1)
            hist = hist.dropna()

            f_f_three_factor_model = reader.DataReader('F-F_Research_Data_Factors', 
                                                    'famafrench', 
                                                    end=f"{data[f'Q({-i})']['date'][0]}-{data[f'Q({-i})']['date'][1]}-01", 
                                                    start=f"{data[f'Q({-i})']['date'][0] - 5}-{data[f'Q({-i})']['date'][1] - 2}-01")

            f_f_momentum_model = reader.DataReader('F-F_Momentum_Factor', 
                                                    'famafrench', 
                                                    end=f"{data[f'Q({-i})']['date'][0]}-{data[f'Q({-i})']['date'][1]}-01", 
                                                    start=f"{data[f'Q({-i})']['date'][0] - 5}-{data[f'Q({-i})']['date'][1] - 2}-01")

        f_f_three_factor_model = f_f_three_factor_model[0]
        f_f_three_factor_model.index = f_f_three_factor_model.index.to_timestamp()

        f_f_momentum_model = f_f_momentum_model[0]
        f_f_momentum_model.index = f_f_momentum_model.index.to_timestamp()

        f_f_models = pd.concat([f_f_three_factor_model, f_f_momentum_model], axis=1)
        f_f_models.columns = [i.replace(' ', '') for i in f_f_models.columns]

        result = pd.concat([hist, f_f_models], axis=1).dropna()
        result['Return - RF'] = result.Return - result.RF

        X_ff = result[['Mkt-RF', 'SMB', 'HML']].to_numpy()
        y_ff = result['Return - RF'].to_numpy()

        reg_ff = LinearRegression().fit(X_ff, y_ff)
        
        result['FF Three Factors Model Return'] = reg_ff.predict(X_ff)
        result['Squared Error'] = (result['Return'] - result['FF Three Factors Model Return']) ** 2

        X_momentum = result[['Mom']].to_numpy()
        y_momentum = result['Return'].to_numpy()

        reg_momentum = LinearRegression().fit(X_momentum, y_momentum)
        
        try:
            data[f'Q({-i})']['cash'] = round(cash_quarter / total_assets_quarter * 100, 3)
        except:
            data[f'Q({-i})']['cash'] = 0
        try:
            data[f'Q({-i})']['stdebt'] = round(short_term_debt_quarter / total_assets_quarter * 100, 3)
        except:
            data[f'Q({-i})']['stdebt'] = 0
        try:
            data[f'Q({-i})']['ltdebt'] = round(long_term_debt_quarter / total_assets_quarter * 100, 3)
        except:
            data[f'Q({-i})']['ltdebt'] = 0
        try:
            data[f'Q({-i})']['mkt'] = round(reg_ff.coef_[0] * 100, 3)
        except:
            data[f'Q({-i})']['mkt'] = 0
        try:
            data[f'Q({-i})']['smb'] = round(reg_ff.coef_[1] * 100, 3)
        except:
            data[f'Q({-i})']['smb'] = 0
        try:
            data[f'Q({-i})']['hml'] = round(reg_ff.coef_[2] * 100, 3)
        except:
            data[f'Q({-i})']['hml'] = 0
        try:
            data[f'Q({-i})']['residuals'] = round(result['Squared Error'].mean() * 100, 3)
        except:
            data[f'Q({-i})']['residuals'] = 0
        try:
            data[f'Q({-i})']['mom'] = round(reg_momentum.coef_[0] * 100, 3)
        except:
            data[f'Q({-i})']['mom'] = 0
    data = pd.DataFrame.from_dict(data, orient='index', columns=['date', 'cash', 'stdebt', 'ltdebt', 'mkt', 'smb', 'hml', 'residuals', 'mom'])
    return data

def get_sector_comparison(ticker):
    """
    return: pandas dataframe or empty dictionary (if company is not found)
        The dataframe consists of summary statistics of the corresponding GICS Sector:
            1. Cash / Total Asset
            2. ShortTermDebt / Total Asset
            3. LongTermDebt / Total Asset
            4. Wavg. of RDSGA / Total Assets
            5. β (Mkt-Rf) (F/F 3-Factor Model)
            6. β (Smb) (F/F 3-Factor Model)
            7. β (Hml) (F/F 3-Factor Model)
            8. Residuals
            9. β (Mom) (F/F Momentum Model)
            10. Score
    
    input:
        ticker (str): company's ticker
    """
    ticker = ticker.upper()
    headers = ['Ticker', 'Security', 'Sector', 'Sub-Industry', 'CIK', 
                'Cash', 'ShortTermDebt', 'LongTermDebt', 'RDSGA', 'Mkt', 'Smb', 'Hml', 'Residuals', 'Mom', 'Score']
    dataframe = pd.read_csv('EdgarData/S&P500.csv', index_col=0)
    dataframe.columns = headers

    data_wanted = dataframe[dataframe['Ticker'] == ticker.strip()].reset_index(drop=True)

    if data_wanted.empty:
        return {}
    data_wanted = data_wanted.to_dict('index') # change dataframe to json data

    sector_dataframe = pd.read_csv(f"EdgarData/S&P500_{data_wanted[0]['Sector'].replace(' ', '_')}.csv", index_col=0) # get the info of corresponding sector
    sector_dataframe.columns = headers
    sector_data_wanted = sector_dataframe[sector_dataframe['Ticker'] == ticker.strip()].reset_index(drop=True)
    if sector_data_wanted.empty:
        return {}
    sector_data_wanted = sector_data_wanted.to_dict('index')
    sector_quantile = sector_dataframe.quantile([.1, .9]).round(3) # get the 10th and 90th percentile
    sector_quantile = sector_quantile.drop(['CIK'], axis=1)
    sector_describe = sector_dataframe.describe().round(3) # get the statistics
    sector_describe = sector_describe.drop(['CIK'], axis=1)
    sector_stat = pd.concat([sector_quantile, sector_describe]) # concatenate together into one dataframe
    
    return sector_stat

def get_report_links(ticker):
    """
    return: dictionary
        The dictionary consists of the link of statements of the company for the latest quarter.
    
    input:
        ticker (str): company's ticker
    """
    def get_master_idx(year, qtr):
        # header for requests url
        headers = {
                    'Host': 'www.sec.gov', 'Connection': 'close',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'X-Requested-With': 'XMLHttpRequest',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'
                }
        
        # index in the master.idx file in one line (format: xxxx|xxxx|xxxx|xxxx|xxxx.txt)
        cik, company_name, form_type, upload_date, url = 0, 1, 2, 3, 4
        
        download = requests.get(f'https://www.sec.gov/Archives/edgar/full-index/{year}/{qtr}/master.idx', headers=headers).content
        
        try:
            download = download.decode("utf-8").split('\n') # encoding
        except:
            return 'Not Found!'

        download_clean = download.copy() # make a copy
        download_clean = [i.split('|') for i in download_clean if '.txt' in i] # clear the details of master.idx file
        # extract only url with form_type of 10-Q and 10-K
        download_clean = [i for i in download_clean if i[form_type] == '10-Q' or i[form_type] == '10-K']
        
        return download_clean # return either 'Not Found !' or correct list
    
    # Function: Getting filingsummary.xml file for the latest quarter
    # The url of the filingsummary.xml file: https://www.sec.gov/Archives/edgar/data/xxxxxx/xxxxxxxxxxxxxxxxxx/FilingSummary.xml

    def get_filingsummary_link(ticker):
        # 'https://www.sec.gov/files/company_tickers_exchange.json': json file of US Stock Exchange listed companies' ticker
        company_json = requests.get('https://www.sec.gov/files/company_tickers_exchange.json').json()

        # make the json file to data frame and extract only the ticker's column and cik's column
        company_dict = pd.DataFrame(company_json['data'], columns=company_json['fields'])
        company_dict = company_dict[['cik', 'ticker']]
        company_dict = company_dict.to_dict('records') # return back to dictionary
        
        # capitalized ticker
        ticker = ticker.upper()
        
        # header for requests url
        headers = {
                    'Host': 'www.sec.gov', 'Connection': 'close',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'X-Requested-With': 'XMLHttpRequest',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'
                }
        
        # initialize variable get_cik
        get_cik = -1

        # index in the master.idx file in one line (format: xxxx|xxxx|xxxx|xxxx|xxxx.txt)
        cik, company_name, form_type, upload_date, url = 0, 1, 2, 3, 4
        
        # initial values
        target = []
        quaters = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
        result = ''
        
        # get today's date
        today = date.today()
        year = today.year
        month = today.month
        
        # check the ticker
        for i in company_dict:
            if i['ticker'] == ticker: # check whether the ticker is in the company_dict
                get_cik = str(i['cik'])

        if get_cik == -1: # ticker doesn't exist in the company_dict
            return 'Not Found!'
        
        # get the latest quarter period
        if month in [1, 2, 3]: # 'QTR1'
            qtr = 0
        elif month in [4, 5, 6]: # 'QTR2'
            qtr = 1
        elif month in [7, 8, 9]: # 'QTR3'
            qtr = 2
        else: # 'QTR4'
            qtr = 3
            
        while True:
            try:
                get_list = get_master_idx(year, quaters[qtr]) # retrieve the latest .txt file of 10-Q or 10-K file
            except:
                return 'Not Found!'
            
            # failed retrive
            if get_list == 'Not Found!':
                return 'Not Found!'

            for item in get_list:
                if item[cik] == get_cik:
                    target = item

            if target != []:
                target_url = target[url] # edgar/data/xxxxxx/xxxxxxxxxx-xx-xxxxxx.txt
                target_url_clean = target_url.replace('-', '') # edgar/data/xxxxxx/xxxxxxxxxxxxxxxxxx.txt
                target_url_clean = target_url_clean.split('.')[0] # edgar/data/xxxxxx/xxxxxxxxxxxxxxxxxx
                
                to_get_html_site = f'https://www.sec.gov/Archives/{target_url}'
                data_report = requests.get(to_get_html_site, headers=headers).content
                data_report = data_report.decode("utf-8") 
                data_report = data_report.split('<FILENAME>')
                #data[1]
                data_report = data_report[1].split('\n')[0]

                result_report = f'https://www.sec.gov/Archives/{target_url_clean}/{data_report}'
                # make url 'https://www.sec.gov/Archives/{target_url_clean}/' a json format
                result = f'https://www.sec.gov/Archives/{target_url_clean}/index.json'

                content = requests.get(result, headers=headers).json() 
                
                for file in content['directory']['item']:
                    # Retrieve the filing summary file, then get the url of the file.
                    if file['name'] == 'FilingSummary.xml': # FilingSummary.xml is standardized
                        filing_summary_url = f"https://www.sec.gov{content['directory']['name']}/{file['name']}"
                        # print(f'url: {filing_summary_url}')
                        # https://www.sec.gov/Archives/edgar/data/xxxxxx/xxxxxxxxxxxxxxxxxx/FilingSummary.xml
                break
            else:
                # Go back to previous quarter
                if qtr == 0:
                    qtr = 3
                    year -= 1
                else:
                    qtr -= 1
                continue
        
        # create a base url for corresponding filing folder
        # https://www.sec.gov/Archives/edgar/data/xxxxxx/xxxxxxxxxxxxxxxxxx/
        try: 
            base = filing_summary_url.replace('FilingSummary.xml', '')
        except:
            return 'Not Found!'

        # parse the content of FilingSummary.xml file
        filing_summary = requests.get(filing_summary_url, headers=headers).content
        filing_summary_soup = bs.BeautifulSoup(filing_summary, 'lxml')

        # retrieve the 'myreports' tag, this tag is standardized, and consists of all the individual reports submitted.
        reports = filing_summary_soup.find('myreports') 

        # create a list to store all the individual reports
        reports_list = []
        reports_list.append(result_report)
        # loop through each report in the 'myreports' tag but avoid the last one as this will cause an error.
        for report in reports.find_all('report')[:-1]:

            # make each of the elements in the reports dictionary
            report_dict = {}
            report_dict['title'] = report.shortname.text.upper() # text of the 'shortname' tag
            report_dict['whole-title'] = report.longname.text.upper() # text of the 'longname' tag
            report_dict['position'] = report.position.text # text of the 'position' tag
            report_dict['url'] = f'{base}{report.htmlfilename.text}' # self-created url
            report_dict['category'] = report.menucategory.text # text of the 'menucategory' tag
            # all tags are standardized
            
            # append the dictionary to the reports.
            reports_list.append(report_dict)

        return reports_list
    
    ticker = ticker.upper()
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    snp500_table = tables[0]
    tickers_list = snp500_table['Symbol'].to_list()
    for i in range(len(tickers_list)):
        tickers_list[i] = tickers_list[i].replace('.', '-')
    ticker = ticker.strip() # clear the spaces
    if ticker not in tickers_list:
        return {}
    all_reports = get_filingsummary_link(ticker)
    # print(isinstance(all_reports, list))

    if isinstance(all_reports, list) == True: # check whether the all_reports is a list
        # create a dictionary which contains only reports categorized to 'Statements'
        statements_dict = {}
        
        statements_dict['Financial Report'] = all_reports[0]
        
        for report_dict in all_reports[1:]: # retrieve the elements of all_reports
            if report_dict['category'] == 'Statements':
                if 'PARENTHETICAL' not in report_dict['title']:
                    statements_dict[report_dict['title']] = report_dict['url'] # {'report-title': 'report-url'}
    else:
        return {}

    return statements_dict

def get_sorted_sector_comparison(ticker, variable="Score"):
    """
    return: pandas dataframe or empty dictionary (if company is not found)
        The dataframe consists of top 10's ranking based on the variable:
    
    input:
        ticker (str): company's ticker
        variable (str): 'Cash', 'RDSGA', 'Score', 'ShortTermDebt', or 'LongTermDebt'. Default: 'Score'
    """
    ticker = ticker.upper()
    if variable not in ['Cash', 'RDSGA', 'Score', 'ShortTermDebt', 'LongTermDebt']:
        return {}
    
    headers = ['Ticker', 'Security', 'Sector', 'Sub-Industry', 'CIK', 
                'Cash', 'ShortTermDebt', 'LongTermDebt', 'RDSGA', 'Mkt', 'Smb', 'Hml', 'Residuals', 'Mom', 'Score']
    dataframe = pd.read_csv('EdgarData/S&P500.csv', index_col=0)
    dataframe.columns = headers

    data_wanted = dataframe[dataframe['Ticker'] == ticker.strip()].reset_index(drop=True)

    if data_wanted.empty:
        return {}
    data_wanted = data_wanted.to_dict('index')

    sector_dataframe = pd.read_csv(f"EdgarData/S&P500_{data_wanted[0]['Sector'].replace(' ', '_')}.csv", index_col=0)
    sector_dataframe.columns = headers
    if variable in ['Cash', 'RDSGA', 'Score']:
        sector_dataframe_wanted = sector_dataframe.sort_values(by=[variable, 'Ticker'], ascending=False).reset_index(drop=True)[['Ticker', 'Security', 'Sub-Industry', variable]]
    else:
        sector_dataframe_wanted = sector_dataframe.sort_values(by=[variable, 'Ticker']).reset_index(drop=True)[['Ticker', 'Security', 'Sub-Industry', variable]]
    return sector_dataframe_wanted