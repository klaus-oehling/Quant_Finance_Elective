import numpy as np
import pandas as pd
import scipy.optimize as opt
from xbbg import blp
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import defaultdict
os.chdir(r'C:\Users\gabrielna\Downloads')


############################################# IMPORTING DATA
tickers = [
    "AAPL US Equity", "AMZN US Equity", "GOOG US Equity", "BRK.A US Equity", "SBUX US Equity",
    "KO US Equity", "DIS US Equity", "FDX US Equity", "LUV US Equity", "GE US Equity",
    "AXP US Equity", "COST US Equity", "NKE US Equity", "BMWYY US Equity", "PG US Equity",
    "IBM US Equity", "JWN US Equity", "SINGY US Equity", "JNJ US Equity", "WFM US Equity",
    "SSNLF US Equity", "MCD US Equity", "MMM US Equity", "MSFT US Equity", "TM US Equity",
    "BA US Equity", "XOM US Equity", "WMT US Equity", "TGT US Equity", "JPM US Equity"]
     
start_date = "2014-12-31"
end_date = "2024-12-31"
   
preco = blp.bdh(tickers=tickers, flds=["PX_LAST"], start_date=start_date, end_date=end_date, adjust="all")
preco.columns = preco.columns.get_level_values(0)
preco.index = pd.to_datetime(preco.index)
preco = preco.dropna(axis=1, how='all')
ret = preco.pct_change()
ret = ret.fillna(0)
ret = ret.iloc[1:,:]

rf = blp.bdh(tickers=['USGG10YR Index'], flds=["PX_LAST"], start_date=start_date, end_date=end_date)
rf.columns = rf.columns.get_level_values(0)
rf.index = pd.to_datetime(rf.index)
rf = rf[rf.index.isin(ret.index)]
day_ret_rf = rf/25200

factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv', skiprows=3)

factors.iloc[:, 0] = pd.to_datetime(factors.iloc[:, 0], format='%Y%m%d')
factors.set_index(factors.iloc[:, 0], inplace=True)
factors.drop(factors.columns[0], axis=1, inplace=True)
factors = factors.loc[start_date:end_date]
factors = factors.astype(float)
factors['Mkt'] = factors['Mkt-RF'] + factors['RF']
factors = factors.drop(columns=['Mkt-RF','RF'])
factors = factors.iloc[1:,:]
factors = factors/100
factors = sm.add_constant(factors)

sp500 = blp.bdh(tickers=['SPX Index'], flds=["PX_LAST"], start_date=start_date, end_date=end_date, adjust="all")
sp500.columns = sp500.columns.get_level_values(0)
sp500.index = pd.to_datetime(sp500.index)
sp500 = sp500.iloc[1:,:]
day_ret_sp500 = sp500.pct_change()


############################################# FUNDAMENTAL VALUES    
r = np.mean(ret.to_numpy(), axis=0, keepdims=True)
num_stocks = len(ret.columns)

returns = ret
daily_ret_rf = day_ret_rf

############################################# FUNCTIONS AND PARAMS
def infos(pesos):
    pesos = np.array(pesos)
    retorno = (r@pesos).reshape(1, 1)[0, 0]
    sig = np.sqrt(pesos.T @ v @ pesos).reshape(1, 1)[0, 0]
    shar = (returns.sub(daily_ret_rf.iloc[:,0], axis=0)*pesos.T).sum(axis=1).mean()/sig
    return {'Return': retorno, 'Risk': sig, 'Sharpe': shar}
def maximizar_sharpe(pesos):  
    return -infos(pesos)['Sharpe']
def minimizar_risco(pesos):  
    return infos(pesos)['Risk']

constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
initializer = num_stocks * [1./num_stocks,]
bounds = tuple((0,1) for x in range(num_stocks))


############################################# COVARIANCE MATRIX USING FACTORS
betas = np.empty((0, len(factors.columns)), float)
variancias_residuos = []
for resposta in ret.columns:
    y = ret[resposta]
    X = factors.copy()
    modelo = sm.OLS(y, X).fit()
    
    betas = np.append(betas, [modelo.params.values], axis=0)
    variancias_residuos.append(modelo.resid.var())
    
# Covariancia entre fatores igual ou diferente de 0. Com covariancia igual a zero, as variancias dos retornos nao batem exatamente com a variancia empirica
# factor_vars = np.diag(factors.var())
factor_vars = np.cov(factors, rowvar=False)
resid_vars = np.diag(variancias_residuos)
v = betas@factor_vars@betas.T + resid_vars
fac_cov = pd.DataFrame(v)
fac_cov.to_csv('fac_cov.csv', index=False, header=False) 

min_var = opt.minimize(minimizar_risco, initializer, method = 'SLSQP', bounds = bounds, constraints = constraints)

pesos_min_var_factors = min_var['x']
weights_with_factors = [f"{valor}%" for valor in np.round(pesos_min_var_factors * 100, 2)]
weights_with_factors = pd.Series(weights_with_factors, index=ret.columns)
dados_port_min_factors = infos(pesos_min_var_factors)


############################################# EMPIRIC COVARIANCE MATRIX
v = np.cov(ret, rowvar=False)
cov = pd.DataFrame(v)
cov.to_csv('cov.csv', index=False, header=False) 

min_var = opt.minimize(minimizar_risco, initializer, method = 'SLSQP', bounds = bounds, constraints = constraints)

pesos_min_var_empiric = min_var['x']
weights_with_empiric = [f"{valor}%" for valor in np.round(pesos_min_var_empiric * 100, 2)]
weights_with_empiric = pd.Series(weights_with_empiric, index=ret.columns)
dados_port_min_empiric = infos(pesos_min_var_empiric)


############################################# PRINTANDO RESULTADOS
print('Weights empiric covariance matrix:')
print(weights_with_empiric)
print('')
print('Weights factor based covariance matrix:')
print(weights_with_factors)
print('')

print('###################### MVP FOR THE EMPIRIC COVARIANCE MATRIX (MONTHLY) ######################')
print('Return =', round(dados_port_min_empiric['Return'], 4), '%')
print('Std deviation =', round(dados_port_min_empiric['Risk'], 4), '%')
print('Sharpe ratio =', round(dados_port_min_empiric['Sharpe'], 4))
print('')

print('###################### MVP FOR THE FACTOR BASED COVARIANCE MATRIX (MONTHLY) ######################')
print('Return =', round(dados_port_min_factors['Return'], 4), '%')
print('Std deviation =', round(dados_port_min_factors['Risk'], 4), '%')
print('Sharpe ratio =', round(dados_port_min_factors['Sharpe'], 4))

plt.plot(pesos_min_var_factors, label='Factor based covariance')
plt.plot(pesos_min_var_empiric, label='Empiric covariance')
plt.title('Weights concentration')
plt.xlabel('Stock')
plt.ylabel('Weight')
plt.legend()


############################################# BACKTEST
results = []
index = []
results_factors = []
   
# FIRST LOOKBACK FACTORS
returns = ret[0:252]
daily_ret_rf = day_ret_rf[0:252] 
    
betas = np.empty((0, len(factors.columns)), float)
variancias_residuos = []
for resposta in returns.columns:
    y = returns[resposta]
    X = factors.iloc[0:252,:].copy()
    modelo = sm.OLS(y, X).fit()
    
    betas = np.append(betas, [modelo.params.values], axis=0)
    variancias_residuos.append(modelo.resid.var())
    
# factor_vars = np.diag(factors.var())
factor_vars = np.cov(factors, rowvar=False)
resid_vars = np.diag(variancias_residuos)

v = betas@factor_vars@betas.T + resid_vars
r = np.mean(returns.to_numpy(), axis=0, keepdims=True)

sharpe_max = opt.minimize(maximizar_sharpe,initializer,method = 'SLSQP',bounds = bounds,constraints = constraints)        
pesos_otimos_factors = sharpe_max['x']


# FIRST LOOKBACK 
v = np.cov(returns, rowvar=False)

sharpe_max = opt.minimize(maximizar_sharpe,initializer,method = 'SLSQP',bounds = bounds,constraints = constraints)        
pesos_otimos = sharpe_max['x']
    
       
# DIVIDING BY MONTH
datas = ret.index
datas_por_mes = defaultdict(list)
for data in datas:
    chave_mes = (data.year, data.month)
    datas_por_mes[chave_mes].append(data)
meses_unicos = datas.to_series().dt.to_period("M").unique().tolist()
   

for meses in meses_unicos[12:]:
    datas_mes = list(datas[datas.to_series().dt.to_period("M") == meses])
    
    for data in datas_mes:
        # RESULT FACTORS
        result_factors = sum(pesos_otimos_factors*ret.loc[data])
        results_factors.append(result_factors)
        
        # RESULT
        result = sum(pesos_otimos*ret.loc[data])
        results.append(result)
        
        index.append(data)
   
    # LOOKBACK FACTORS
    lista_meses = pd.date_range((meses-11).start_time, (meses+1).start_time, freq='M').to_period('M')
    returns = ret[pd.to_datetime(ret.index).to_period('M').isin(lista_meses)]
    fac = factors[pd.to_datetime(factors.index).to_period('M').isin(lista_meses)]
    daily_ret_rf = day_ret_rf[pd.to_datetime(day_ret_rf.index).to_period('M').isin(lista_meses)]

    betas = np.empty((0, len(fac.columns)), float)
    variancias_residuos = []
    for resposta in returns.columns:
        modelo = sm.OLS(returns[resposta], fac).fit()
        betas = np.append(betas, [modelo.params.values], axis=0)
        variancias_residuos.append(modelo.resid.var())
    # factor_vars = np.diag(factors.var())
    factor_vars = np.cov(fac, rowvar=False)
    resid_vars = np.diag(variancias_residuos)
    
    v = betas@factor_vars@betas.T + resid_vars
    r = np.mean(returns.to_numpy(), axis=0, keepdims=True)

    sharpe_max_factors = opt.minimize(maximizar_sharpe,initializer,method = 'SLSQP',bounds = bounds,constraints = constraints)        
    pesos_otimos_factors = sharpe_max_factors['x']

    # LOOKBACK
    v = np.cov(returns, rowvar=False)

    sharpe_max = opt.minimize(maximizar_sharpe,initializer,method = 'SLSQP',bounds = bounds,constraints = constraints)        
    pesos_otimos = sharpe_max['x']
    
    print(f'rebalanceamento {meses}')

results_factors = pd.DataFrame(results_factors, index=index)
results = pd.DataFrame(results, index=index)

results_cum = (1+results).cumprod()
results_factors_cum = (1+results_factors).cumprod()
sp500_cum = (1+day_ret_sp500).cumprod()
rf_cum = (1+day_ret_rf).cumprod()



plt.plot(results_factors_cum, label='Factors covariance matrix max sharpe')
plt.plot(results_cum, label='Empiric covariance matrix max sharpe')
plt.plot(sp500_cum, label='S&P 500')
plt.plot(rf_cum, label='Risk-Free')
   
plt.title('All cumulative returns')
plt.xlabel('Date')
plt.ylabel('Cumulative returns')
plt.legend()
        
