import pandas as pd
import pandas_datareader.data as reader
from datetime import date
from yahooquery import Ticker
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import time
import os

def update_csv():
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    snp500_table = tables[0]
    tickers_list = snp500_table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'CIK']].values.tolist()
    for i in range(len(tickers_list)):
        tickers_list[i][0] = tickers_list[i][0].replace('.', '-')

    datas = []

    f_f_three_factor_model = reader.DataReader('F-F_Research_Data_Factors', 
                                            'famafrench', 
                                            end=f"{date.today().year}-{date.today().month}-01", 
                                            start=f"{date.today().year - 5}-{date.today().month - 2}-01")

    f_f_momentum_model = reader.DataReader('F-F_Momentum_Factor', 
                                            'famafrench', 
                                            end=f"{date.today().year}-{date.today().month}-01", 
                                            start=f"{date.today().year - 5}-{date.today().month - 2}-01")

    f_f_three_factor_model = f_f_three_factor_model[0]
    f_f_three_factor_model.index = f_f_three_factor_model.index.to_timestamp()

    f_f_momentum_model = f_f_momentum_model[0]
    f_f_momentum_model.index = f_f_momentum_model.index.to_timestamp()

    f_f_models = pd.concat([f_f_three_factor_model, f_f_momentum_model], axis=1)
    f_f_models.columns = [i.replace(' ', '') for i in f_f_models.columns]

    for ticker in tickers_list:
        value = Ticker(ticker[0])
        element = []
        element.append(ticker)
        element = [item for sublist in element for item in sublist]
        financials_annual = value.income_statement(frequency='a', trailing=False)
        financials_annual = financials_annual.sort_values(by=['asOfDate'], ascending=False).T
        financials_annual.columns = [f"{financials_annual.columns[i]}_{i}" for i in range(len(financials_annual.columns))]
        balance_sheet_annual = value.balance_sheet(frequency='a', trailing=False).T.iloc[:, -1]
        balance_sheet_quarter = value.balance_sheet(frequency='q', trailing=False).T.iloc[:, -1]
        
        weight_avg_RDSGA_annual = 0
        coefficient = [1, 0.75, 0.5, 0.25]
        for i in range(len(coefficient)):
            try:
                if financials_annual.loc['ResearchAndDevelopment', financials_annual.columns[i]] == None or np.isnan(financials_annual.loc['ResearchAndDevelopment', financials_annual.columns[i]]):
                    r_and_d_annual = 0 
                else:
                    r_and_d_annual = financials_annual.loc['ResearchAndDevelopment', financials_annual.columns[i]]
            except:
                r_and_d_annual = 0

            try:
                if financials_annual.loc['SellingGeneralAndAdministration', financials_annual.columns[i]] == None or np.isnan(financials_annual.loc['SellingGeneralAndAdministration', financials_annual.columns[i]]):
                    marketing_expenses_annual = 0 
                else:
                    marketing_expenses_annual = financials_annual.loc['SellingGeneralAndAdministration', financials_annual.columns[i]]
            except:
                marketing_expenses_annual = 0
            numerator = (r_and_d_annual + marketing_expenses_annual / 3) * coefficient[i]
            weight_avg_RDSGA_annual += numerator

        try:
            if balance_sheet_annual['TotalAssets'] == None or np.isnan(balance_sheet_annual['TotalAssets']):
                total_assets_annual = 0 
            else:
                total_assets_annual = balance_sheet_annual['TotalAssets']
        except:
            total_assets_annual = 0

        try:
            if balance_sheet_quarter['TotalAssets'] == None or np.isnan(balance_sheet_quarter['TotalAssets']):
                total_assets_quarter = 0 
            else:
                total_assets_quarter = balance_sheet_quarter['TotalAssets']
        except:
            total_assets_quarter = 0

        try:
            if balance_sheet_quarter['CashAndCashEquivalents'] == None or np.isnan(balance_sheet_quarter['CashAndCashEquivalents']):
                cash_quarter = 0 
            else:
                cash_quarter = balance_sheet_quarter['CashAndCashEquivalents']
        except:
            cash_quarter = 0

        try:
            if balance_sheet_quarter['CurrentDebt'] == None or np.isnan(balance_sheet_quarter['CurrentDebt']):
                short_term_debt_quarter = 0 
            else:
                short_term_debt_quarter = balance_sheet_quarter['CurrentDebt']
        except:
            short_term_debt_quarter = 0

        try:
            if balance_sheet_quarter['LongTermDebt'] == None or np.isnan(balance_sheet_quarter['LongTermDebt']):
                long_term_debt_quarter = 0 
            else:
                long_term_debt_quarter = balance_sheet_quarter['LongTermDebt']
        except:
            long_term_debt_quarter = 0
        
        try:
            cash_per_assets = round(cash_quarter / total_assets_quarter * 100, 3)
        except:
            cash_per_assets = np.nan
        element.append(cash_per_assets)
        
        try:
            st_debt_per_assets = round(short_term_debt_quarter / total_assets_quarter * 100, 3)
        except:
            st_debt_per_assets = np.nan
        element.append(st_debt_per_assets)
        
        try:
            lt_debt_per_assets = round(long_term_debt_quarter / total_assets_quarter * 100, 3)
        except:
            lt_debt_per_assets = np.nan
        element.append(lt_debt_per_assets)
        
        try:
            weighted_avg_RDSGA_per_assets = round(weight_avg_RDSGA_annual / total_assets_annual * 100, 3)
        except:
            weighted_avg_RDSGA_per_assets = np.nan
        element.append(weighted_avg_RDSGA_per_assets)
        
        value_hist = yf.Ticker(ticker[0])
        hist = value_hist.history(interval="1mo", 
                                end=f"{date.today().year}-{date.today().month}-01", 
                                start=f"{date.today().year - 5}-{date.today().month - 2}-01")
        hist = hist.dropna()
        hist = hist[['Close']]
        hist['Return'] = hist['Close'].rolling(window=2).apply(lambda x: x[1]/x[0] - 1)
        hist = hist.dropna()
        
        result = pd.concat([hist, f_f_models], axis=1).dropna()

        result['Return - RF'] = result.Return - result.RF
        
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
        
        X_momentum = result[['Mom']].to_numpy()
        y_momentum = result['Return'].to_numpy()

        reg_momentum = LinearRegression().fit(X_momentum, y_momentum)
        slope_MOM = round(reg_momentum.coef_[0] * 100, 3)
        element.append(slope_MOM)
        
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
        time.sleep(3)

    try:
        os.makedirs('EdgarData')
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

    dataframe = pd.DataFrame(data=datas, columns=headers)
    dataframe = (dataframe
                .sort_values(by=['GICS Sector', 'Ticker', 'GICS Sub-Industry'], ascending=True, na_position='first')
                .reset_index(drop=True))
    
    dataframe = dataframe.dropna()
    dataframe.to_csv('EdgarData/S&P500.csv')
    
    sector_list = dataframe['GICS Sector'].unique().tolist()
    
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
        sectors[i]['data'].to_csv(f"EdgarData/S&P500_{sectors[i]['sector'].replace(' ', '_')}.csv")
    
    return 'Done'

def get_company_details(ticker):
    headers = ['Ticker', 'Security', 'Sector', 'Sub-Industry', 'CIK', 
                'Cash', 'ShortTermDebt', 'LongTermDebt', 'RDSGA', 'Mkt', 'Smb', 'Hml', 'Residuals', 'Mom', 'Score']
    dataframe = pd.read_csv('EdgarData/S&P500.csv', index_col=0)
    dataframe.columns = headers

    data_wanted = dataframe[dataframe['Ticker'] == ticker.strip()].reset_index(drop=True)
    data_wanted_value = data_wanted[[i for i in data_wanted.columns if i not in ['Ticker', 'Security', 'Sector', 'Sub-Industry', 'CIK']]]

    if data_wanted.empty:
        return {}
    data_wanted = data_wanted.to_dict('index')

    sector_dataframe = pd.read_csv(f"EdgarData/S&P500_{data_wanted[0]['Sector'].replace(' ', '_')}.csv", index_col=0)
    sector_dataframe.columns = headers
    sector_data_wanted = sector_dataframe[sector_dataframe['Ticker'] == ticker.strip()].reset_index(drop=True)
    if sector_data_wanted.empty:
        return {}
    sector_data_wanted = sector_data_wanted.to_dict('index')
    sector_quantile = sector_dataframe.quantile([.1, .9]).round(3)
    sector_quantile = sector_quantile.drop(['CIK'], axis=1)
    sector_describe = sector_dataframe.describe().round(3)
    sector_describe = sector_describe.drop(['CIK'], axis=1)
    sector_stat = pd.concat([sector_quantile, sector_describe])
    
    data_upload = {}
    data_upload['overall'] = dataframe[['Cash', 'ShortTermDebt', 'LongTermDebt', 'RDSGA', 'Mkt', 'Smb', 'Hml', 'Residuals', 'Mom', 'Score']].describe().T[['mean', '50%']].T.round(3)
    data_upload['company'] = data_wanted
    data_upload['sector_stat'] = sector_stat
    data_upload['company_value'] = data_wanted_value
    
    return data_upload

def get_overall_rank(value):
    headers = ['Ticker', 'Security', 'Sector', 'Sub-Industry', 'CIK', 
                'Cash', 'ShortTermDebt', 'LongTermDebt', 'RDSGA', 'Mkt', 'Smb', 'Hml', 'Residuals', 'Mom', 'Score']
    dataframe = pd.read_csv(f"EdgarData/S&P500.csv", index_col=0)
    dataframe.columns = headers
    if int(value) >= len(dataframe):
        dataframe = dataframe.sort_values(by=['Score', 'Ticker'], ascending=False).reset_index(drop=True)[['Ticker', 'Security', 'Sector', 'Score']]
    else:
        dataframe = dataframe.sort_values(by=['Score', 'Ticker'], ascending=False).reset_index(drop=True)[['Ticker', 'Security', 'Sector', 'Score']][:int(value)]
    result = dataframe
    return result

def get_additional_analytics(tickers):
    # print(request.data)
    if isinstance(tickers, str):
        tickers = [tickers]

    # print(ticker_list)
    data = {}

    for i in tickers:
        data[i] = {}
        ticker_hist = yf.Ticker(i)
        
        hist_1y = ticker_hist.history(period='1y')
        hist_1y = hist_1y.dropna()
        hist_1y = hist_1y['Close']

        hist_3mo = ticker_hist.history(period='3mo')
        hist_3mo = hist_3mo.dropna()
        hist_3mo = hist_3mo['Close']

        hist_30d = ticker_hist.history(period='30d')
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
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    snp500_table = tables[0]
    tickers_list = snp500_table['Symbol'].to_list()
    for i in range(len(tickers_list)):
        tickers_list[i] = tickers_list[i].replace('.', '-')

    if symbol not in tickers_list:
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
