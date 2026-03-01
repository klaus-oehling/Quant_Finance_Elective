import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
from xbbg import blp
from matplotlib.collections import LineCollection


############################################# IMPORTING DATA
tickers = [
    "AAPL US Equity", "AMZN US Equity", "GOOG US Equity", "BRK.A US Equity", "SBUX US Equity",
    "KO US Equity", "DIS US Equity", "FDX US Equity", "LUV US Equity", "GE US Equity",
    "AXP US Equity", "COST US Equity", "NKE US Equity", "BMWYY US Equity", "PG US Equity",
    "IBM US Equity", "JWN US Equity", "SINGY US Equity", "JNJ US Equity", "WFM US Equity",
    "SSNLF US Equity", "MCD US Equity", "MMM US Equity", "MSFT US Equity", "TM US Equity",
    "BA US Equity", "XOM US Equity", "WMT US Equity", "TGT US Equity", "JPM US Equity"]
     
start_date = "2015-01-01"
end_date = "2024-12-31"
   
preco = blp.bdh(tickers=tickers, flds=["PX_LAST"], start_date=start_date, end_date=end_date, adjust="all")
preco.columns = preco.columns.get_level_values(0)
preco.index = pd.to_datetime(preco.index)
preco = preco.dropna(axis=1, how='all')
preco = preco.resample('M').last()
ret = preco.pct_change()
ret = ret.fillna(0)

rf = blp.bdh(tickers=['USGG10YR Index'], flds=["PX_LAST"], start_date=start_date, end_date=end_date)
rf.columns = rf.columns.get_level_values(0)
rf.index = pd.to_datetime(rf.index)
month_ret_rf = rf/1200
month_ret_rf = month_ret_rf.resample('M').last()
month_ret_rf = month_ret_rf['USGG10YR Index']


############################################# FUNDAMENTAL VALUES    
r = np.mean(ret.to_numpy(), axis=0, keepdims=True)
v = np.cov(ret, rowvar=False)
c = np.corrcoef(ret, rowvar=False)
num_stocks = len(ret.columns)


############################################# POINTS CLOUD
mu=[]
sigma=[]
sharpe=[]
for i in range(10000):
    pesos = np.random.dirichlet(np.ones(num_stocks), size=1).T
    mu.append((r@pesos)[0,0])
    sigma.append(np.sqrt(pesos.T@v@pesos)[0,0])
    sharpe.append((ret.sub(month_ret_rf, axis=0)*pesos.T).sum(axis=1).mean()/sigma[i])

   
dados=pd.concat([pd.DataFrame(mu),pd.DataFrame(sigma),pd.DataFrame(sharpe)], axis=1)
dados.columns=['Mean','Standard Deviation', 'Sharpe']


############################################# FUNCTIONS
def infos(pesos):
    pesos = np.array(pesos)
    retorno = (r@pesos).reshape(1, 1)[0, 0]
    sig = np.sqrt(pesos.T @ v @ pesos).reshape(1, 1)[0, 0]
    shar = (ret.sub(month_ret_rf, axis=0)*pesos.T).sum(axis=1).mean()/sig
    return {'Return': retorno, 'Risk': sig, 'Sharpe': shar}
def maximizar_sharpe(pesos):  
    return -infos(pesos)['Sharpe']
def minimizar_risco(pesos):  
    return infos(pesos)['Risk']


############################################# MAXIMIZING SHARPE AND MINIMIZING RISK
constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
initializer = num_stocks * [1./num_stocks,]
bounds = tuple((0,1) for x in range(num_stocks))
sharpe_max=opt.minimize(maximizar_sharpe,
                                initializer,
                                method = 'SLSQP',
                                bounds = bounds,
                                constraints = constraints)
pesos_otimos=sharpe_max['x']
dados_port_otimo=infos(pesos_otimos)

min_var=opt.minimize(minimizar_risco,
                            initializer,
                            method = 'SLSQP',
                            bounds = bounds,
                            constraints = constraints)
pesos_min_var=min_var['x']
dados_port_min=infos(pesos_min_var)


############################################# EFICIENT FRONTIER CALCULUS
intervalo_retornos = np.linspace(np.min(r),
                                np.max(r),70)
fronteira = []
pesos_fronteira=[]
for intervalo_retorno in intervalo_retornos:
    constraints = ({'type':'eq','fun': lambda x: infos(x)['Return']-intervalo_retorno},
                {'type':'eq','fun': lambda x: np.sum(x)-1})
    fronteira_valores = opt.minimize(minimizar_risco,
                        initializer,
                        method = 'SLSQP',
                        bounds = bounds,
                        constraints = constraints)
    pesos_fronteira.append(fronteira_valores['x'])

fronteira = pd.DataFrame(columns=['Return', 'Risk', 'Sharpe'])
for i in range(len(intervalo_retornos)):
    novos_dados = infos(pesos_fronteira[i])
    novos_dados_df = pd.DataFrame([novos_dados])
    fronteira = pd.concat([fronteira, novos_dados_df], ignore_index=True)


############################################# CAPITAL ALOCATION LINE
sig=np.arange(0,dados_port_otimo['Risk'],0.0001)
medi=[]
for i in range(0,len(sig)):
    medi.append(-sharpe_max['fun']*sig[i] + month_ret_rf.mean())


############################################# DRAWING GRAPHS
# Colorando a fronteira eficiente
x = fronteira['Risk'].values
y = fronteira['Return'].values
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(fronteira['Sharpe'].min(), fronteira['Sharpe'].max())
cmap = plt.get_cmap('binary')
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(fronteira['Sharpe'].values)
lc.set_linewidth(2)
fig, ax = plt.subplots()

# Adicionar a fronteira
ax.add_collection(lc)
ax.autoscale_view()
ax.set_xlim(0, fronteira['Risk'].max())
ax.set_ylim(0, fronteira['Return'].max())

# Adicionar outros elementos
plt.plot(sig, medi, color='black')
plt.scatter(dados['Standard Deviation'], dados['Mean'], c=dados['Sharpe'], cmap='binary', s=1)
plt.scatter(dados_port_otimo['Risk'], dados_port_otimo['Return'], color='gold', s=100, marker='*', label='Max Sharpe Portfolio')  # Ponto ótimo
plt.scatter(dados_port_min['Risk'], dados_port_min['Return'], color='red', s=100, marker='*', label='Min Variance Portfolio')  # Ponto mínimo

# Adicionar legenda e rótulos
plt.xlabel('Standard deviation')
plt.ylabel('Return')
plt.title('Eficient frontier')
plt.legend()
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label('Sharpe Ratio')


############################################# PRINTANDO RESULTADOS
print('MAXIMUM SHARPE PORTFOLIO RESULTS')
print(dados_port_otimo)
print('MINIMUM VARIANCE PORTFOLIO RESULTS')
print(dados_port_min)


############################################# EFICIENT FRONTIER FOR VARIOUS CORRELATIONS
lista_espacada = np.linspace(-1, 1, 7).tolist()
num_stocks = 2
initializer = num_stocks * [1./num_stocks,]
bounds = tuple((0,1) for x in range(num_stocks))
constraints = ({'type':'eq','fun': lambda x: infos(x)['Return']-intervalo_retorno},
            {'type':'eq','fun': lambda x: np.sum(x)-1})
plt.figure()
        
for k in lista_espacada:

    # Parâmetros
    r = np.array([[0.005], [0.02]]).T
    sigma_x = 0.01
    sigma_y = 0.02
    c = k
    cov_xy = c * sigma_x * sigma_y
    v = np.asmatrix([[sigma_x**2, cov_xy], [cov_xy, sigma_y**2]])

    # Gerando os dados
    n = 120
    ret = pd.DataFrame(np.random.multivariate_normal(list(r[0,:]), v, size=n))
    correlacao = np.corrcoef(ret.iloc[:,0], ret.iloc[:,1])[0, 1]
   
    # Cálculo da fronteira
    intervalo_retornos = np.linspace(np.min(r),np.max(r),70)
    fronteira = []
    pesos_fronteira=[]
    for intervalo_retorno in intervalo_retornos:
        fronteira_valores = opt.minimize(minimizar_risco,
                            initializer,
                            method = 'SLSQP',
                            bounds = bounds,
                            constraints = constraints)
        pesos_fronteira.append(fronteira_valores['x'])
   
    fronteira = pd.DataFrame(columns=['Return', 'Risk', 'Sharpe'])
    for i in range(len(intervalo_retornos)):
        novos_dados = infos(pesos_fronteira[i])
        novos_dados_df = pd.DataFrame([novos_dados])
        fronteira = pd.concat([fronteira, novos_dados_df], ignore_index=True)
   
    # Gráfico
    plt.plot(fronteira['Risk'],fronteira['Return'])
    plt.text(fronteira['Risk'].iloc[34]-0.0005, fronteira['Return'].iloc[34], f'{round(correlacao,2)}')
   
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.2)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.2)
plt.xlim(-0.001, 0.021)
plt.ylim(-0.001, 0.021)  
plt.xlabel('Standard deviation')
plt.ylabel('Return')
plt.title('Eficient frontier on diferent correlations')