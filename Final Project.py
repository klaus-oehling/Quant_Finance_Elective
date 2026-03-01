import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

################################## DATA ##################################
ret = pd.read_excel('dados.xlsx', sheet_name='ret diario')
ret = ret.rename(columns={'Unnamed: 0': 'Date'})
ret = ret.set_index('Date', drop=True)
ret = ret.reindex(pd.date_range(start='2015-01-02', end='2024-12-31', freq='B'))
ret = ret.dropna()

factors = pd.read_excel('fatores_macro.xlsx', sheet_name='dados diarios')
factors = factors.fillna(method='ffill')
factors = factors.iloc[:,[0,1,3,6,7]]
factors = factors.rename(columns={'Unnamed: 0': 'Date','CPI YOY Index': 'BC', 'GT2 Govt': 'MP', 'SPX Index': 'RS', 'USTBTOT  Index': 'IT'})
factors = factors.set_index('Date', drop=True)
factors = factors.reindex(pd.date_range(start='2015-01-02', end='2024-12-31', freq='B'))
factors = factors.loc[ret.index,:]

factors['BC'] = factors['BC'] - factors['BC'].shift(252)
factors['IT'] = factors['IT'] - factors['IT'].shift(252)
factors['MP'] = factors['MP'] - factors['MP'].shift(252)
factors['RS'] = (factors['RS'] / factors['RS'].shift(252)) - 1
factors = factors.dropna()

ret = ret.loc[factors.index,:]

rf = pd.read_excel('fatores_macro.xlsx', sheet_name='dados diarios')
rf = rf.rename(columns={'Unnamed: 0': 'Date','FEDL01 Index': 'RF'})
rf = rf.iloc[:,[0,10]]
rf = rf.set_index('Date', drop=True)

SPX = pd.read_excel('fatores_macro.xlsx', sheet_name='dados diarios')
SPX = SPX.rename(columns={'Unnamed: 0': 'Date','FEDL01 Index': 'RF'})
SPX = SPX.iloc[:,[0,6]]
SPX = SPX.set_index('Date', drop=True)
SPX = SPX.pct_change()

GVT10 = pd.read_excel('fatores_macro.xlsx', sheet_name='dados diarios')
GVT10 = GVT10.rename(columns={'Unnamed: 0': 'Date','FEDL01 Index': 'RF'})
GVT10 = GVT10.iloc[:,[0,5]]
GVT10 = GVT10.set_index('Date', drop=True)

################################## REGRESSION ##################################
coefs = pd.DataFrame(index=ret.columns, columns=factors.columns, dtype=float)

# Para cada ação (coluna de ret), faz a regressão contra os fatores
for col in ret.columns:
    y = ret[col].astype(float)
    X = factors.astype(float)
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    coefs.loc[col] = model.params[factors.columns]


BC = []
MP = []
IT = []
RS = []
comp = []
LSR_weights = []
DIR_weights = []

for i in ret.loc['2016-02':].index:

    start = (i - pd.DateOffset(months=1)).replace(day=1)
    start2 = ret[(ret.index.month == i.month) & (ret.index.year == i.year)].index[0]

    end = (start + pd.offsets.MonthEnd(0))

    signal = factors.loc[start:end].mean()
    signal = signal*coefs

    LSR_pesos = pd.Series(0, index=ret.columns)
    DIR_pesos = pd.Series(0, index=ret.columns)
    for j in signal.columns:

        # LONG-SHORT by rank (market neutral)
        specific_signal = pd.DataFrame(signal[j].sort_values(ascending=True))

        rank = pd.Series(range(1, 30))
        media = rank.mean()
        desvio_padrao = rank.std()
        rank = (rank - media) / desvio_padrao
        rank.index = specific_signal.index

        rank = rank / (rank[rank>0].sum()/2)
        rank = rank.reindex(ret.columns)

        if start2 == i:
            LSR = sum(ret.loc[i]*rank)
        else:
            rank = rank*(1+ret.loc[start2:i]).cumprod().iloc[-1,:]
            LSR = (ret.loc[i]*rank).values.sum()
        LSR_pesos = LSR_pesos + rank

        # DIRECTIONAL
        desvio_padrao = specific_signal.std()
        w = specific_signal/ desvio_padrao
        if desvio_padrao[0] == 0:
            w = pd.Series(0, index=specific_signal.index)

        if float(w[w>0].sum()/2) > float(abs(w[w<0].sum()/2)):
            w = w / (w[w>0].sum()/2)
        else:
            w = w / abs(w[w<0].sum()/2)

        w = w.reindex(ret.columns).squeeze()

        if start2 == i:
            DIR = sum(ret.loc[i]*w)
        else:
            w = w*(1+ret.loc[start2:i]).cumprod().iloc[-1,:]
            DIR = (ret.loc[i]*w).values.sum()
        DIR_pesos = DIR_pesos + w

        if j == 'BC':
            BC.append([i,LSR, DIR])
        elif j == 'MP':
            MP.append([i,LSR, DIR])
        elif j == 'IT':
            IT.append([i,LSR, DIR])
        elif j == 'RS':
            RS.append([i,LSR, DIR])

    LSR_weights.append(LSR_pesos)
    DIR_weights.append(DIR_pesos)


BC = pd.DataFrame(BC)
BC = BC.set_index(BC.iloc[:,0])
BC = BC.drop(BC.columns[0], axis=1)
BC = BC.rename(columns={1: 'LSR', 2: 'DIR'})
BC = BC.sum(axis=1)/2

IT = pd.DataFrame(IT)
IT = IT.set_index(IT.iloc[:,0])
IT = IT.drop(IT.columns[0], axis=1)
IT = IT.rename(columns={1: 'LSR', 2: 'DIR'})
IT = IT.sum(axis=1)/2

MP = pd.DataFrame(MP)
MP = MP.set_index(MP.iloc[:,0])
MP = MP.drop(MP.columns[0], axis=1)
MP = MP.rename(columns={1: 'LSR', 2: 'DIR'})
MP = MP.sum(axis=1)/2

RS = pd.DataFrame(RS)
RS = RS.set_index(RS.iloc[:,0])
RS = RS.drop(RS.columns[0], axis=1)
RS = RS.rename(columns={1: 'LSR', 2: 'DIR'})
RS = RS.sum(axis=1)/2

COMPOSITE = (BC + IT + MP + RS)/4
CHOSEN = (BC + RS)/2

LSR_weights = pd.DataFrame(LSR_weights)/4
LSR_weights = LSR_weights.set_index(ret.loc['2016-02':].index)

DIR_weights = pd.DataFrame(DIR_weights)/4
DIR_weights = DIR_weights.set_index(ret.loc['2016-02':].index)

LSR_alocacao = LSR_weights.sum(axis=1)
DIR_alocacao = DIR_weights.sum(axis=1)
alocacao = (LSR_alocacao + DIR_alocacao)/2

BCa = (1+BC).cumprod()
ITa = (1+IT).cumprod()
MPa = (1+MP).cumprod()
RSa = (1+RS).cumprod()
rfa = (1+rf).cumprod()
COMPOSITEa = (1+COMPOSITE).cumprod()
CHOSENa = (1+CHOSEN).cumprod()

plt.plot(COMPOSITE)
################################## SHARPES ##################################
rf = rf.loc[COMPOSITE.index,:]/100
rf = (1+rf)**(1/252)-1
GVT10 = GVT10.loc[COMPOSITE.index,:]/100
GVT10 = (1+GVT10)**(1/252)-1
SPX = SPX.loc[COMPOSITE.index,:]
SPX_ex = SPX.subtract(rf.iloc[:,0], axis = 0)

ex = COMPOSITE.subtract(rf.iloc[:,0], axis = 0)
sm.OLS(ex, sm.add_constant(SPX_ex), missing='drop').fit().params
ex.corr(SPX_ex.iloc[:,0])
ex.corr(GVT10.iloc[:,0])
ex = (1+ex.mean())**252-1
sharpe_c = ex/(COMPOSITE.std()*np.sqrt(252))

ex = BC.subtract(rf.iloc[:,0], axis = 0)
sm.OLS(ex, sm.add_constant(SPX_ex), missing='drop').fit().params
ex.corr(SPX_ex.iloc[:,0])
ex.corr(GVT10.iloc[:,0])
ex = (1+ex.mean())**252-1
sharpe_bc = ex/(BC.std()*np.sqrt(252))

ex = IT.subtract(rf.iloc[:,0], axis = 0)
sm.OLS(ex, sm.add_constant(SPX_ex), missing='drop').fit().params
ex.corr(SPX_ex.iloc[:,0])
ex.corr(GVT10.iloc[:,0])
ex = (1+ex.mean())**252-1
sharpe_it = ex/(IT.std()*np.sqrt(252))

ex = MP.subtract(rf.iloc[:,0], axis = 0)
sm.OLS(ex, sm.add_constant(SPX_ex), missing='drop').fit().params
ex.corr(SPX_ex.iloc[:,0])
ex.corr(GVT10.iloc[:,0])
ex = (1+ex.mean())**252-1
sharpe_mp = ex/(MP.std()*np.sqrt(252))

ex = RS.subtract(rf.iloc[:,0], axis = 0)
sm.OLS(ex, sm.add_constant(SPX_ex), missing='drop').fit().params
ex.corr(SPX_ex.iloc[:,0])
ex.corr(GVT10.iloc[:,0])
ex = (1+ex.mean())**252-1
sharpe_rs = ex/(RS.std()*np.sqrt(252))

ex = CHOSEN.subtract(rf.iloc[:,0], axis = 0)
sm.OLS(ex, sm.add_constant(SPX_ex), missing='drop').fit().params
ex.corr(SPX_ex.iloc[:,0])
ex.corr(GVT10.iloc[:,0])
ex = (1+ex.mean())**252-1
sharpe_chosen = ex/(CHOSEN.std()*np.sqrt(252))

sharpes = []
betas = []
data = []
ex = CHOSEN.subtract(rf.iloc[:,0], axis = 0)
for i in CHOSEN.loc['2017-2':].index:
    betas.append(sm.OLS(ex[(i - pd.DateOffset(years=1)):i], sm.add_constant(SPX_ex[(i - pd.DateOffset(years=1)):i]), missing='drop').fit().params[1])
    sharpes.append(((1+ex[(i - pd.DateOffset(years=1)):i].mean())**252-1)/(CHOSEN[(i - pd.DateOffset(years=1)):i].std()*np.sqrt(252)))
    data.append(i)

df = pd.DataFrame({'sharpe': sharpes, 'beta': betas}, index=data)


################################## DRWADOWNS ##################################
drawdown_SPX = ((1+SPX).cumprod()-(1+SPX).cumprod().cummax()) / (1+SPX).cumprod().cummax()
drawdown_COMPOSITE = ((1+COMPOSITE).cumprod()-(1+COMPOSITE).cumprod().cummax()) / (1+COMPOSITE).cumprod().cummax()

plt.plot(drawdown_SPX)
plt.plot(drawdown_COMPOSITE)

min(drawdown_SPX[1:].values)
min(drawdown_COMPOSITE[1:].values)

