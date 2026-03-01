############### LIBRARIES ###############
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_ccf

############### DATA ###############
ret_diario = pd.read_excel('dados.xlsx', sheet_name='ret diario')
ret_diario = ret_diario.set_index(ret_diario.iloc[:,0])
ret_diario = ret_diario.iloc[:,1:]
ret = ret_diario.iloc[:,0:2]

############### CROSS-CORRELOGRAM ###############
plot_ccf(ret.iloc[:,0], ret.iloc[:,1], lags=20)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.title("Cross-correlation between Apple and lagged Amazon returns")

plot_ccf(ret.iloc[:,1], ret.iloc[:,0], lags=20)
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.title("Cross-correlation between Amazon and lagged Apple returns")

############### VAR1 ###############
model = VAR(ret)
results = model.fit(1)
print(results.summary())

