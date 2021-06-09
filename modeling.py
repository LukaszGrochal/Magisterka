import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from format_data import data_from_2007_to_2011, data_from_2015_to_now
from format_data import tickers
from draw_plots import zwrot_logarytmiczny
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from arch.univariate import Normal, GeneralizedError



dane_nowe = data_from_2015_to_now()


# load statsmodels
import statsmodels.tsa.arima.model as  stm
# fit ARIMA model
model = stm.ARIMA(dane_nowe["KGH"].logarytmiczna_stopa_zwrotu, order=(1,0,0))
arima_model = model.fit()
# one-step out-of sample forecast
print(arima_model.resid[:10])
arima_residuals = arima_model.resid


# fit a GARCH(1,1) model on the residuals of the ARIMA model
garch = arch_model(arima_residuals, p=1, q=1)

garch_fitted = garch.fit()


# Use ARIMA to predict mu
predicted_mu = arima_model.forecast(steps=10) #predict(n_periods=1)[0]
# Use GARCH to predict the residual
print(predicted_mu)
garch_forecast = garch_fitted.forecast(horizon=10)
print(f'predicted_mu: \n{predicted_mu}\n')
print('garch_forecast: \n ', garch_forecast.mean, '\n')
print (garch_forecast.variance)
predicted_et = garch_forecast.variance
# Combine both models' output: yt = mu + et
prediction = predicted_mu + predicted_et
print(prediction)