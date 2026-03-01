############################# BASICS #############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

############################# DATA #############################
df = pd.read_excel("QF_Project_8.xlsx")
df = df.iloc[1:,:]
df = df.set_index('Unnamed: 0')
df = df.iloc[:,1:]
df = df.rename(columns={df.columns[0]: 'Log Ratio', df.columns[1]: 'Return'})

df['Log Ratio'] = np.log(df['Log Ratio'])
df['Annual Return'] = (1 + df['Return']).rolling(252).apply(lambda x: np.prod(x) - 1)

df = df.iloc[252:,:]

############################# IN SAMPLE FORECASTING #############################
df_reg = df.copy()
df_reg['Log_Ratio_Lag1'] = df_reg['Log Ratio'].shift(1)
df_reg = df_reg.dropna()

X = df_reg['Log_Ratio_Lag1']
y = df_reg['Annual Return']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})
df['In Sample Forecast'] = (model.params[0] + model.params[1]*df['Log Ratio']).shift(1)

############################# OUT OF SAMPLE FORECASTING #############################
OSF = []
data = []
total = len(df_reg.index[3:])
for i, data in enumerate(df_reg.index[3:]):

    if data != df_reg.index[3]:
        OSF.append([forecast,data])

    dfa = df_reg.loc[:data]

    X = dfa['Log_Ratio_Lag1']
    y = dfa['Annual Return']
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':1})

    forecast = (model.params[0] + model.params[1]*dfa['Log Ratio'].loc[data])

    percent = int((i + 1) / total * 100)
    barras = int(percent / 5)
    print(f"\rProgresso: |{'#' * barras}{' ' * (20 - barras)}| {percent}% concluído", end='')

OSF = pd.DataFrame(OSF)
OSF = OSF.set_index(1)
OSF = OSF.rename(columns={0: 'Out of Sample Forecast'})

df = df.join(OSF)
df['OSF error'] = df['Annual Return'] - df['Out of Sample Forecast']
df['ISF error'] = df['Annual Return'] - df['In Sample Forecast']
MSE_OSF = mean_squared_error(df.dropna()['Annual Return'], df.dropna()['Out of Sample Forecast'])
std_error_osf = df.dropna()['OSF error'].std()

MSE_ISF = mean_squared_error(df.dropna()['Annual Return'], df.dropna()['In Sample Forecast'])
std_error_isf = df.dropna()['ISF error'].std()

############################# GRAPH #############################
plt.figure(figsize=(12, 6))
# Traçar cada série com cores específicas
plt.plot(df.index, df['Annual Return'], label='S&P500 Annual Return', color='black', linewidth=1)
plt.plot(df.index, df['In Sample Forecast'], label='In-Sample Forecast', linewidth=1)
plt.plot(df.index, df['Out of Sample Forecast'], label='Out-of-Sample Forecast', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
# Legenda, título e eixos
plt.legend()
plt.title('Annual Return vs Forecasts')
plt.xlabel('Date')
plt.ylabel('Return')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
# Traçar cada série
plt.plot(df.index, df['ISF error'], label=f'In-Sample Forecast error; MSE = {round(MSE_ISF,5)}; ST-ERROR = {round(std_error_isf,4)}', linewidth=1)
plt.plot(df.index, df['OSF error'], label=f'Out-of-Sample Forecast error; MSE = {round(MSE_OSF,5)}; ST-ERROR = {round(std_error_osf,4)}', linewidth=1)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
# Legenda, título e eixos
plt.legend()
plt.title('Forecast Errors')
plt.xlabel('Date')
plt.ylabel('Error')
plt.grid(True)
plt.tight_layout()
plt.show()