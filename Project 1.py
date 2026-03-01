import pandas as pd
import random
import numpy as np
from collections import defaultdict
from xbbg import blp
import matplotlib.pyplot as plt

def portfolio(k = None, simulation = False, tickers = None):
############################### OFICIAL
   
    if not simulation:
        ticker = tickers
    else:
        sp500_data = blp.bds("SPX Index", "INDX_MEMBERS")
        random_tickers = random.sample(list(sp500_data.iloc[:,0]), k)
        random_tickers = [ticker.replace(" UW","") for ticker in random_tickers]
        random_tickers = [ticker.replace(" UN","") for ticker in random_tickers]
        ticker = [ticker + " US Equity" for ticker in random_tickers]
       
    start_date = "2014-12-31"
    end_date = "2024-12-31"
   
    marketcap = blp.bdh(tickers=ticker, flds=["CUR_MKT_CAP"], start_date=start_date, end_date=end_date, adjust="all")
    marketcap.columns = marketcap.columns.get_level_values(0)
    marketcap.index = pd.to_datetime(marketcap.index)
    marketcap = marketcap.dropna(axis=1)
   
    if not simulation:
        preco = blp.bdh(tickers=ticker, flds=["PX_LAST"], start_date=start_date, end_date=end_date, adjust="all")
        preco.columns = preco.columns.get_level_values(0)
        preco.index = pd.to_datetime(preco.index)
        preco = preco.dropna(axis=1)
        ret = preco.pct_change()
       
        sp500 = blp.bdh(tickers=['SPX Index'], flds=["PX_LAST"], start_date=start_date, end_date=end_date, adjust="all")
        sp500.columns = sp500.columns.get_level_values(0)
        sp500.index = pd.to_datetime(sp500.index)
        day_ret_sp500 = sp500.pct_change()
       
        rf = blp.bdh(tickers=['USGG10YR Index'], flds=["PX_LAST"], start_date=start_date, end_date=end_date)
        rf.columns = rf.columns.get_level_values(0)
        rf.index = pd.to_datetime(rf.index)
        day_ret_rf = rf[1:]/36000
        month_ret_rf = rf[1:]/1200
   
   
    ################################################## PARTE FEITA
    vw = []
    ew = []
    index = []
    rp_vw = []
    rp_ew = []
    rp_sp500 = []
    month_ret_ew_list=[]
    month_ret_vw_list=[]
    month_ret_sp500_list=[]
    er_ew = []
    er_vw = []
   
    data = marketcap.index[0]
   
    # FIRST LOOKBACK EQUAL WEIGHT
    parcela_marketcap_ew = 1/marketcap.loc[data]
   
    # FIRST LOOKBACK VALUE WEIGHT
    pesos=marketcap.loc[data]/marketcap.loc[data].sum()
    parcela_marketcap_vw = (pesos*len(marketcap.columns))/marketcap.loc[data]
   
    # POSITION EQUAL WEIGHT
    posicao_ew = sum(parcela_marketcap_ew*marketcap.loc[data])
    ew.append(posicao_ew)
       
    # POSITION VALUE WEIGHT
    posicao_vw = sum(parcela_marketcap_vw*marketcap.loc[data])
    vw.append(posicao_vw)
       
    index.append(data)
   
    last_rebal_ew = parcela_marketcap_ew
    last_rebal_vw = parcela_marketcap_vw
    total_purchases_ew = 0
    total_sales_ew = 0
    total_purchases_vw = 0
    total_sales_vw = 0
   
           
    # DIVIDING BY MONTH
    datas = marketcap.index[1:]
    datas_por_mes = defaultdict(list)
    for data in datas:
        chave_mes = (data.year, data.month)
        datas_por_mes[chave_mes].append(data)
    meses_unicos = datas.to_series().dt.to_period("M").unique().tolist()
   
   
   
   
    for meses in meses_unicos:
        datas_mes = list(datas[datas.to_series().dt.to_period("M") == meses])
       
        for d in range(len(datas_mes)):
            if not simulation and d == 0:
                # TURNOVER EW
                values_ew = (parcela_marketcap_ew - last_rebal_ew)*marketcap.loc[data]
                purchases_ew = sum(value for value in values_ew if value > 0)
                sales_ew = sum(value for value in values_ew if value < 0)
               
                # TURNOVER VW
                values_vw = (parcela_marketcap_vw - last_rebal_vw)*marketcap.loc[data]
                purchases_vw = sum(value for value in values_vw if value > 0)
                sales_vw = sum(value for value in values_vw if value < 0)
               
        if not simulation:      
            last_posicao_ew = posicao_ew
            last_posicao_vw = posicao_vw
            data_anterior = data
       
        for data in datas_mes:
           
            # POSITION EQUAL WEIGHT
            posicao_ew = sum(parcela_marketcap_ew*marketcap.loc[data])
            ew.append(posicao_ew)
           
            # POSITION VALUE WEIGHT
            posicao_vw = sum(parcela_marketcap_vw*marketcap.loc[data])
            vw.append(posicao_vw)
           
            index.append(data)
       
        # ALL OPERATIONS (TURNOVER CALC)
        if not simulation:
            last_rebal_ew = parcela_marketcap_ew
            last_rebal_vw = parcela_marketcap_vw
            total_purchases_ew = purchases_ew + total_purchases_ew
            total_sales_ew = sales_ew + total_sales_ew
            total_purchases_vw = purchases_vw + total_purchases_vw
            total_sales_vw = sales_vw + total_sales_vw
   
        # LOOKBACK EQUAL WEIGHT - LAST DAY OF THE MONTH
        parcela_marketcap_ew = posicao_ew/(len(marketcap.columns)*marketcap.loc[data])
       
        # LOOKBACK VALUE WEIGHT - LAST DAY OF THE MONTH
        pesos=marketcap.loc[data]/marketcap.loc[data].sum()
        parcela_marketcap_vw = (pesos*posicao_vw)/marketcap.loc[data]
       
        # RISK PREMIUM
        if not simulation:
            month_ret_sp500 = (sp500.loc[data] - sp500.loc[data_anterior])/sp500.loc[data_anterior]
            month_ret_sp500_list.append(month_ret_sp500)            
            rp_sp500.append(month_ret_sp500[0] - month_ret_rf.loc[data])
       
            month_ret_ew = (posicao_ew-last_posicao_ew)/last_posicao_ew
            month_ret_ew_list.append(month_ret_ew)
            rp_ew.append(month_ret_ew - month_ret_rf.loc[data])
            er_ew.append(month_ret_ew - month_ret_sp500[0])
   
            month_ret_vw = (posicao_vw-last_posicao_vw)/last_posicao_vw
            month_ret_vw_list.append(month_ret_vw)
            rp_vw.append(month_ret_vw - month_ret_rf.loc[data])
            er_vw.append(month_ret_vw - month_ret_sp500[0])

   
    ew_trajetoria = pd.DataFrame(ew, index=index)
    ew_trajetoria = ew_trajetoria.rename(columns={0: 'Equal Weight'})
   
    vw_trajetoria = pd.DataFrame(vw, index=index)
    vw_trajetoria = vw_trajetoria.rename(columns={0: 'Value Weight'})
   
    posicoes = pd.concat([ew_trajetoria,vw_trajetoria], axis = 1)
   
    day_ret_ew = posicoes['Equal Weight'].pct_change()
    day_ret_vw = posicoes['Value Weight'].pct_change()
   
   
    if simulation:
        global simulations
        global var_ew
        global var_vw
        print('K =', k, 'Simulation number', j+1)
       
    simulations.append(k)
    var_ew.append(day_ret_ew.var())
    var_vw.append(day_ret_vw.var())
   
   
    if not simulation:
        ###################### METRICAS
        turnover_ew = min(total_purchases_ew, abs(total_sales_ew))/((ew[-1]-ew[0])/2)*100
        turnover_vw = min(total_purchases_vw, abs(total_sales_vw))/((vw[-1]-vw[0])/2)*100
       
        ew_cum = (1+day_ret_ew).cumprod()
        vw_cum = (1+day_ret_vw).cumprod()
        sp500_cum = (1+day_ret_sp500).cumprod()
        rf_cum = (1+day_ret_rf).cumprod()
     
        ann_avg_ret_ew = ((1 + day_ret_ew.mean())**252-1)
        ann_avg_ret_vw = ((1 + day_ret_vw.mean())**252-1)
        ann_avg_ret_sp500 = ((1 + day_ret_sp500.mean())**252-1)
       
        ann_std_ew = day_ret_ew.std()*np.sqrt(252)
        ann_std_vw = day_ret_vw.std()*np.sqrt(252)
        ann_std_sp500 = day_ret_sp500.std()*np.sqrt(252)
       
        sharpe_ew = ((1 + np.array(rp_ew)).prod()**(1/len(rp_ew))-1)/np.std(month_ret_ew_list)
        sharpe_vw = ((1 + np.array(rp_vw)).prod()**(1/len(rp_vw))-1)/np.std(month_ret_vw_list)
        sharpe_sp500 = ((1 + np.array(rp_sp500)).prod()**(1/len(rp_sp500))-1)/np.std(month_ret_sp500_list)
       
        ar_ew = np.mean(er_ew)
        ar_vw = np.mean(er_vw)
        te_ew = np.std(er_ew)
        te_vw = np.std(er_vw)
       
        information_ew = ar_ew/te_ew
        information_vw = ar_vw/te_vw
       
       
        ######################## REPORT
        print(40*'#', 'Report', 40*'#')
       
        print('')
       
        print('Daily return for Equal Weight Portfolio:')
        print('')
        print(day_ret_ew)
       
        print('')
       
        print('Daily return for Value Weight Portfolio:')
        print('')
        print(day_ret_vw)
       
        print('')
       
        print('Equal Weight Portfolio metrics (SP500 in parentheses):')
        print('')
        print('Annualized average return', round(ann_avg_ret_ew*100,2),'% (', round(ann_avg_ret_sp500.iloc[0]*100,2), '% )')
        print('Annualized standard deviation', round(ann_std_ew*100,2),'% (', round(ann_std_sp500.iloc[0]*100,2), '% )')
        print('Sharpe ratio', round(sharpe_ew,2),'(', round(sharpe_sp500,2), ')')
        print('Information ratio', round(information_ew,2))
        print('Turnover',  round(turnover_ew,2), '%')

       
        print('')
       
        print('Value Weight Portfolio metrics (SP500 in parentheses):')
        print('')
        print('Annualized average return', round(ann_avg_ret_vw*100,2),'% (', round(ann_avg_ret_sp500.iloc[0]*100,2), '% )')
        print('Annualized standard deviation', round(ann_std_vw*100,2),'% (', round(ann_std_sp500.iloc[0]*100,2), '% )')
        print('Sharpe ratio', round(sharpe_vw,2),'(', round(sharpe_sp500,2), ')')
        print('Information ratio', round(information_vw,2))
        print('Turnover', round(turnover_vw,2), '%')
     
       
        plt.plot(ew_cum, label='EW Portfolio')
        plt.plot(vw_cum, label='VW Portfolio')
        plt.plot(sp500_cum, label='S&P 500')
        plt.plot(rf_cum, label='Risk-Free')
       
        plt.title('All cumulative returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative returns')
        plt.legend()
       
########################## RUNNING SIMULATIONS
var_ew=[]
var_vw=[]
simulations=[]
for k in range(1,31):
    for j in range(10):
        portfolio(k=k, simulation = True)
frontier = pd.concat([pd.Series(simulations),pd.Series(var_ew),pd.Series(var_vw)], axis = 1)

plt.figure()
plt.scatter(frontier.iloc[:,0], frontier.iloc[:,1])
plt.title('Daily returns Variance: Equal Weight')
plt.xlabel('Number of stocks')
plt.ylabel('Variance')

plt.figure()
plt.scatter(frontier.iloc[:,0], frontier.iloc[:,2])
plt.title('Daily returns Variance: Value Weight')
plt.xlabel('Number of stocks')
plt.ylabel('Variance')

####################### RUNNING 30 SELECTED STOCKS
tickers = [
    "AAPL US Equity", "AMZN US Equity", "GOOG US Equity", "BRK.A US Equity", "SBUX US Equity",
    "KO US Equity", "DIS US Equity", "FDX US Equity", "LUV US Equity", "GE US Equity",
    "AXP US Equity", "COST US Equity", "NKE US Equity", "BMWYY US Equity", "PG US Equity",
    "IBM US Equity", "JWN US Equity", "SINGY US Equity", "JNJ US Equity", "WFM US Equity",
    "SSNLF US Equity", "MCD US Equity", "MMM US Equity", "MSFT US Equity", "TM US Equity",
    "BA US Equity", "XOM US Equity", "WMT US Equity", "TGT US Equity", "JPM US Equity"]

portfolio(simulation = False, tickers = tickers)