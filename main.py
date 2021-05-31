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

if __name__ == '__main__':
    # dane_stare = data_from_2007_to_2011()
    dane_nowe = data_from_2015_to_now()
    print(dane_nowe)

    for ticker in tickers:
    # ticker = 'KGH'
    # print(f'ticker = {ticker} ',acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu))
    # print(f'ticker = {ticker} ', acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, lags=40))
    # print(f'ticker = {ticker} ', acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu,  return_df=True))

        plt.plot(dane_nowe[ticker].zwykla_stopa_zwrotu, label=ticker)
        plt.savefig(f'plots/{ticker}/zwrot_2015_now.png', format='png', dpi=600)
        plt.clf()
        zwrot_logarytmiczny(ticker, dane_nowe[ticker])
        plt.savefig(f'plots/{ticker}/zwrot_log_2015_now.png', format='png', dpi=600)
        plot_pacf(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, lags=40, zero=False)
        plt.savefig(f'plots/{ticker}/plot_pacf_2015_now.png', format='png', dpi=600)
        plot_acf(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, lags=40, zero=False)
        plt.savefig(f'plots/{ticker}/plot_acf_2015_now.png', format='png', dpi=600)
        fig = sm.qqplot(dane_nowe[ticker].logarytmiczna_stopa_zwrotu,  fit=True,  line="s")
        plt.savefig(f'plots/{ticker}/qqplot_2015_now.png', format='png', dpi=600)
        print(f'Test Ljung-Box na logarytmicznych stopach zwrotu dla {ticker}')
        print(f'ticker = {ticker} ', acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, return_df=True))

        print(f'\n\n\nmodelowanie\n')



    # am1 = arch_model(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, mean='AR', lags=1, p=1, o=0, q=1)
    # am1.distribution = Normal()
    #
    # res1 = am1.fit()
    # print(res1.summary())
    # res1.plot()
    # plt.show()
    # mean='Constant'
        arch1 = arch_model(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, mean='AR', lags=1, p=1, o=0, q=1)
        arch1.distribution = GeneralizedError()
        res2 = arch1.fit()
        print(res2.summary())
        # print(am2.pa)
        res2.plot()
        plt.savefig(f'plots/{ticker}/garch_2015_now.png', format='png', dpi=600)
        plt.clf()
    # Plot model fitting results
        plt.plot(res2.conditional_volatility, color='red', label='garch')
        plt.plot(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, color='grey', label='Daily Log Returns', alpha=0.4)
        plt.legend(loc='upper right')
        plt.savefig(f'plots/{ticker}/garch_vs_daily_2015_now.png', format='png', dpi=600)
        plt.clf()
    # res2.hedgehog_plot()
    # print(res2.optimization_result)
    # plt.show()
    # sim_forecasts = res1.forecast(start="2021-4-1", method="simulation", horizon=10, reindex=False)

    # print(sim_forecasts.residual_variance.dropna().head())
    # print(sim_forecasts.simulations.values[:,:2,:])
    # plt.plot(sim_forecasts.simulations.values, label=ticker)
    # plt.show()
    # forecasts = res1.forecast(start="2021-4-1", horizon=5, reindex=False)
    # print(sim_forecasts.simulations.values[0,0:3,:])
    # # sim_forecasts.variance.plot()
    # plt.plot(sim_forecasts.simulations.values[0, 1, :10])
    # plt.plot(sim_forecasts.simulations.values[0, 0, :10])
    # plt.plot(sim_forecasts.simulations.values[0, 2, :10])
    # plt.show()