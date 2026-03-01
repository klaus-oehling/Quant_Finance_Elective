import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from matplotlib.collections import LineCollection
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import numpy as np

import os

diret = "C:\\Users\\gabri\\Downloads"
os.chdir(diret)

############################################# IMPORTING DATA
tickers = [
    "AAPL US Equity", "AMZN US Equity", "GOOG US Equity", "BRK.A US Equity", "SBUX US Equity",
    "KO US Equity", "DIS US Equity", "FDX US Equity", "LUV US Equity", "GE US Equity",
    "AXP US Equity", "COST US Equity", "NKE US Equity", "PEP US Equity", "PG US Equity",
    "IBM US Equity", "JWN US Equity", "AZN US Equity", "JNJ US Equity", "CVX US Equity",
    "TMUS US Equity", "MCD US Equity", "MMM US Equity", "MSFT US Equity", "TM US Equity",
    "BA US Equity", "XOM US Equity", "WMT US Equity", "TGT US Equity", "JPM US Equity"]
     
ret_daily = pd.read_excel('dados.xlsx',sheet_name='ret diario')
ret_monthly = pd.read_excel('dados.xlsx', sheet_name='ret mensal')

price_daily = pd.read_excel('dados.xlsx',sheet_name='preco diario')
price_monthly = pd.read_excel('dados.xlsx',sheet_name='preco mensal')

ret_monthly=ret_monthly.set_index(ret_monthly['Unnamed: 0'])
ret_monthly = ret_monthly.iloc[:,1:]

lags = np.arange(0,21,1)

#####FACP AND ACF RETURNS
figsize = (14, 8)

for ticker in ret_daily.columns:  
    if ticker in ret_monthly.columns:  
        fig, axes = plt.subplots(2, 2, figsize=figsize)  
        
        fig.suptitle(f'{ticker} - PACF and ACF for Daily and Monthly Returns', fontweight='bold', fontsize=14)

       
        plot_pacf(ret_daily[ticker], lags=lags, alpha=0.1, ax=axes[0, 0])
        axes[0, 0].set_title('PACF - Daily Returns')
        axes[0, 0].set_xlabel('Lags')
        axes[0, 0].set_ylabel('PACF')

        plot_acf(ret_daily[ticker], lags=lags, alpha=0.1, ax=axes[0, 1])
        axes[0, 1].set_title('ACF - Daily Returns')
        axes[0, 1].set_xlabel('Lags')
        axes[0, 1].set_ylabel('ACF')

        
        plot_pacf(ret_monthly[ticker], lags=lags, alpha=0.1, ax=axes[1, 0])
        axes[1, 0].set_title('PACF - Monthly Returns')
        axes[1, 0].set_xlabel('Lags')
        axes[1, 0].set_ylabel('PACF')

        plot_acf(ret_monthly[ticker], lags=lags, alpha=0.1, ax=axes[1, 1])
        axes[1, 1].set_title('ACF - Monthly Returns')
        axes[1, 1].set_xlabel('Lags')
        axes[1, 1].set_ylabel('ACF')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
for ticker in price_daily.columns:  
    if ticker in price_monthly.columns:  
        fig, axes = plt.subplots(2, 2, figsize=figsize)  
        
        fig.suptitle(f'{ticker} - PACF and ACF for Daily and Monthly Prices', fontweight='bold', fontsize=14)

       
        plot_pacf(price_daily[ticker], lags=lags, alpha=0.1, ax=axes[0, 0])
        axes[0, 0].set_title('PACF - Daily Prices')
        axes[0, 0].set_xlabel('Lags')
        axes[0, 0].set_ylabel('PACF')

        plot_acf(price_daily[ticker], lags=lags, alpha=0.1, ax=axes[0, 1])
        axes[0, 1].set_title('ACF - Daily Prices')
        axes[0, 1].set_xlabel('Lags')
        axes[0, 1].set_ylabel('ACF')

        
        plot_pacf(price_monthly[ticker], lags=lags, alpha=0.1, ax=axes[1, 0])
        axes[1, 0].set_title('PACF - Monthly Prices')
        axes[1, 0].set_xlabel('Lags')
        axes[1, 0].set_ylabel('PACF')

        plot_acf(price_monthly[ticker], lags=lags, alpha=0.1, ax=axes[1, 1])
        axes[1, 1].set_title('ACF - Monthly Prices')
        axes[1, 1].set_xlabel('Lags')
        axes[1, 1].set_ylabel('ACF')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
    

