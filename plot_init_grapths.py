from format_data import data_from_2007_to_2011, data_from_2015_to_now
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from format_data import tickers
from draw_plots import zwrot_logarytmiczny, zwrot_zwykly, ceny_akcji, statystyki_opisowe, trzy_wykresy, acf_pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import scipy.stats as stats
from datetime import datetime
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox

for ticker in tickers:
    dane_nowe = data_from_2015_to_now()
    print(f'ticker = {ticker} ',acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu))
    print(f'ticker = {ticker} ', acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, lags=40))
    print(f'ticker = {ticker} ', acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu,  return_df=True))

    ceny_akcji(ticker, dane_nowe[ticker])
    plt.savefig(f'plots/{ticker}/ceny_akcji_2015_now.png', format='png', dpi=600)
    plt.clf()
    zwrot_zwykly(ticker, dane_nowe[ticker])
    plt.savefig(f'plots/{ticker}/zwrot_2015_now.png', format='png', dpi=600)
    plt.clf()
    zwrot_logarytmiczny(ticker, dane_nowe[ticker])
    plt.savefig(f'plots/{ticker}/zwrot_log_2015_now.png', format='png', dpi=600)
    plt.clf()

    statystyki_opisowe(ticker, dane_nowe[ticker][:datetime.strptime('2021-05-01', '%Y-%m-%d')]
                       .logarytmiczna_stopa_zwrotu)

    trzy_wykresy(ticker, dane_nowe[ticker])
    plt.savefig(f'plots/{ticker}/trzy_wykresy_w_jednym.png', format='png', dpi=600)
    plt.clf()
    print(f'Test Ljung-Box na logarytmicznych stopach zwrotu dla {ticker}')
    print(f'ticker = {ticker} \n', acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu,
                                                  lags=10, return_df=True))
    print(f'Test Ljung-Box na kwadratach logarytmicznych stopach zwrotu dla {ticker}')
    print(f'ticker = {ticker} \n', acorr_ljungbox(dane_nowe[ticker].logarytmiczna_stopa_zwrotu**2, return_df=True))

    acf_pacf(ticker, dane_nowe[ticker].logarytmiczna_stopa_zwrotu)
    plt.savefig(f'plots/{ticker}/acf_pacf.png', format='png', dpi=600)
    # plt.show()
    plt.clf()
    x = dane_nowe[ticker].logarytmiczna_stopa_zwrotu.autocorr(lag=1)
    print(f'\n{ticker} autokorelacja dla opóźnienia 1 {x}')

    r, p = stats.pearsonr(dane_nowe[ticker].logarytmiczna_stopa_zwrotu[:-1], dane_nowe[ticker].logarytmiczna_stopa_zwrotu[1:])
    print(f'\n\nautokorelacja = {r:.6f} p = {p:.6f}')
    # pd.plotting.autocorrelation_plot(dane_nowe[ticker].logarytmiczna_stopa_zwrotu)
    # plt.show()
    # pd.Series.autocorr(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, lag=1)

    acf_pacf(ticker, dane_nowe[ticker].logarytmiczna_stopa_zwrotu**2)
    plt.savefig(f'plots/{ticker}/acf_pacf_dla_kwadratow.png', format='png', dpi=600)
    # plt.show()
    plt.clf()

    for x in range(1, 20):

        wartosc, p_wartosc, _, _ = het_arch(dane_nowe[ticker].logarytmiczna_stopa_zwrotu, nlags=x)
        print(ticker, f"  dla opóźnienia {x}: wartość = {wartosc:.2} ; p-wartość = {p_wartosc:.2E}")

