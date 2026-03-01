############################## ESSENTIALS ##############################
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
os.chdir(r'C:\Users\oehli\OneDrive - Insper - Instituto de Ensino e Pesquisa\6 SEM\Quantitative Finance')

############################## DATA ##############################
ret_diario = pd.read_excel('dados.xlsx', sheet_name='ret diario')
ret_diario = ret_diario.set_index(ret_diario.iloc[:,0])
ret_diario = ret_diario.iloc[:,1:]

############################## FUNCTIONS ##############################
#def calcular_momentum(retornos):
#    betas = {}
#    for coluna in retornos.columns:
#        X = sm.add_constant(np.array(retornos[coluna][:-1]))
#        y = np.array(retornos[coluna][1:])
#        model = sm.OLS(y, X).fit()
#        betas[coluna] = model.params[1]  # Armazena o coeficiente da regressão para cada ação
#    return betas

for l in [1, 3, 6, 12]:
    lookback_period = l
    
    ############################## DIVIDING BY MONTH ##############################
    datas = ret_diario.index
    datas_por_mes = defaultdict(list)
    for data in datas:
        chave_mes = (data.year, data.month)
        datas_por_mes[chave_mes].append(data)
    meses_unicos_totais = datas.to_series().dt.to_period("M").unique().tolist()
    meses_unicos = meses_unicos_totais[lookback_period:]

    ############################## FIRST LOOKBACK ##############################
    lookback_inicial = meses_unicos_totais[:lookback_period]
    lookback = pd.concat([ret_diario[ret_diario.index.to_series().dt.to_period("M") == mes] for mes in lookback_inicial])

    ############################## ITERATION ##############################
    retorno_ts_ls = []
    retorno_ts_lo = []
    retorno_cs_ls = []
    retorno_cs_lo = []
    retorno_ew_lo = []
    index = []
    for i, meses in enumerate(meses_unicos):
        datas_mes = list(datas[datas.to_series().dt.to_period("M") == meses])
                
        for data in datas_mes:
            #momentum = calcular_momentum(lookback)
            momentum = (1 + lookback).cumprod()
            momentum = momentum.iloc[-1] - 1
            
            # TIME SERIES LONG-SHORT
            momentum_ts_ls = {acao: 1 if mom > 0 else -1 for acao, mom in momentum.items()}
            ret = ret_diario.loc[data] * pd.Series(momentum_ts_ls)
            ret = ret[ret != 0].mean()
            retorno_ts_ls.append(ret)

            # TIME SERIES LONG-ONLY
            momentum_ts_lo = {acao: 1 if mom > 0 else 0 for acao, mom in momentum.items()}
            ret = ret_diario.loc[data] * pd.Series(momentum_ts_lo)
            ret = ret[ret != 0].mean()
            retorno_ts_lo.append(ret)

            # CROSS SECTIONAL LONG-SHORT
            momentum_cs_ls = pd.qcut(pd.Series(momentum), q=3, labels=[-1, 0, 1]).to_dict()
            ret = ret_diario.loc[data] * pd.Series(momentum_cs_ls)
            ret = ret[ret != 0].mean()
            retorno_cs_ls.append(ret)

            # CROSS SECTIONAL LONG-ONLY
            momentum_cs_lo = pd.qcut(pd.Series(momentum), q=3, labels=[0, 1, 2]).map({0: 0, 1: 0, 2: 1}).to_dict()
            ret = ret_diario.loc[data] * pd.Series(momentum_cs_lo)
            ret = ret[ret != 0].mean()
            retorno_cs_lo.append(ret)

            # EW LONG-ONLY
            ret = np.mean(ret_diario.loc[data])
            retorno_ew_lo.append(ret)

            index.append(data)

        lookback_inicial = meses_unicos_totais[i + 1 : i + 1 + lookback_period]
        lookback = pd.concat([ret_diario[ret_diario.index.to_series().dt.to_period("M") == mes] for mes in lookback_inicial])

    ############################## OUTPUT STRUCTURING ##############################
    retorno_df = pd.DataFrame({
        'Momentum Time Series Long-Short': retorno_ts_ls,
        'Momentum Time Series Long-Only': retorno_ts_lo,
        'Momentum Cross Sectional Long-Short': retorno_cs_ls,
        'Momentum Cross Sectional Long-Only': retorno_cs_lo,
        'Long-Only Equall Weight': retorno_ew_lo
    }, index=index)
    retorno_acumulado = (1 + retorno_df).cumprod()

    ############################## METRICS ##############################
    return_ts_ls = pd.Series(retorno_ts_ls)
    return_ts_lo = pd.Series(retorno_ts_lo)
    ann_avg_ret_ts_ls = ((1 + return_ts_ls.mean())**252 - 1)
    ann_avg_ret_ts_lo = ((1 + return_ts_lo.mean())**252 - 1)
    ann_std_ts_ls = return_ts_ls.std() * np.sqrt(252)
    ann_std_ts_lo = return_ts_lo.std() * np.sqrt(252)

    return_cs_ls = pd.Series(retorno_cs_ls)
    return_cs_lo = pd.Series(retorno_cs_lo)
    ann_avg_ret_cs_ls = ((1 + return_cs_ls.mean())**252 - 1)
    ann_avg_ret_cs_lo = ((1 + return_cs_lo.mean())**252 - 1)
    ann_std_cs_ls = return_cs_ls.std() * np.sqrt(252)
    ann_std_cs_lo = return_cs_lo.std() * np.sqrt(252)

    return_ew_lo = pd.Series(retorno_ew_lo)
    ann_avg_ret_ew_lo = ((1 + return_ew_lo.mean())**252 - 1)
    ann_std_ew_lo = return_ew_lo.std() * np.sqrt(252)
    print(f'########### Lookback de {lookback_period} meses finalizado ###########')

    ############################## PLOT ##############################
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(retorno_acumulado['Momentum Cross Sectional Long-Only'], label=f'Momentum Cross Sectional Long-Only\nAAR: {ann_avg_ret_cs_lo:.2%}, ASTD: {ann_std_cs_lo:.2%}, RVR: {ann_avg_ret_cs_lo/ann_std_cs_lo:.2}')
    ax.plot(retorno_acumulado['Momentum Time Series Long-Only'], label=f'Momentum Time Series Long-Only\nAAR: {ann_avg_ret_ts_lo:.2%}, ASTD: {ann_std_ts_lo:.2%}, RVR: {ann_avg_ret_ts_lo/ann_std_ts_lo:.2}')
    ax.plot(retorno_acumulado['Long-Only Equall Weight'], label=f'Long-Only Equall Weight\nAAR: {ann_avg_ret_ew_lo:.2%}, ASTD: {ann_std_ew_lo:.2%}, RVR: {ann_avg_ret_ew_lo/ann_std_ew_lo:.2}')
    ax.plot(retorno_acumulado['Momentum Cross Sectional Long-Short'], label=f'Momentum Cross Sectional Long-Short\nAAR: {ann_avg_ret_cs_ls:.2%}, ASTD: {ann_std_cs_ls:.2%}, RVR: {ann_avg_ret_cs_ls/ann_std_cs_ls:.2}')
    ax.plot(retorno_acumulado['Momentum Time Series Long-Short'], label=f'Momentum Time Series Long-Short\nAAR: {ann_avg_ret_ts_ls:.2%}, ASTD: {ann_std_ts_ls:.2%}, RVR: {ann_avg_ret_ts_ls/ann_std_ts_ls:.2}')
    ax.text(0.05, 0.05, f"Lookback de {lookback_period} meses", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    ax.set_title('PnLs for times series and cross sectional momentum strategies')
    ax.set_xlabel('Date')
    ax.set_ylabel('PnL')
    ax.legend()
    ax.grid(True)