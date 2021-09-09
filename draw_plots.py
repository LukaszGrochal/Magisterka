import matplotlib.pyplot as plt
from scipy.stats import describe
from statistics import median, stdev
from statsmodels.distributions.empirical_distribution import ECDF
def ceny_akcji(ticker, data):
    plt.plot(data['Close'], label=ticker)
    plt.legend(loc='best')


def zwrot_logarytmiczny(ticker, data):
    plt.plot(data['logarytmiczna_stopa_zwrotu'], label=ticker)
    plt.legend(loc='best')


def zwrot_zwykly(ticker, data):
    plt.plot(data['zwykla_stopa_zwrotu'], label=ticker)
    plt.legend(loc='best')


def statystyki_opisowe(ticker, data):
    nobs, minmax, mean, variance, skew, kurtosis = describe(data)
    s_d = stdev(data)
    v_median = median(data)
    v_max = max(data)
    v_min = min(data)
    print(f'indeks = {ticker}, liczba obserwacji = {nobs}, średnia = {mean:>6.5f}, mediana = {v_median:>6.5f},'
          f' maximum = {minmax[1]:>6.5f}, minimum = {minmax[0]:>6.5f}, wariancja = {variance:>6.5f},'
          f' odchylenie standardowe = {s_d:>6.5f}, kurtoza = {kurtosis:>6.5f}, skośność = {skew:>6.5f}')
    # return mean, s_d, variance, kurtosis, skew, v_max, v_min, v_median


def trzy_wykresy(ticker, data):
    """
    tutaj wyświetlamy jeden wykres z trzema sub - wykresami
    wykres logarytmicznych stóp zwrotu,
    zawierał histogram,empiryczną funkcję gęstości,funkcję gęstości rozkładu normalnego z empiryczną średnią i wariancją
    """
    import numpy as np
    from scipy.stats import norm, probplot

    fig, axs = plt.subplots(3, figsize=(5, 8))


    """logarytmiczne stopy zwrotu"""
    axs[0].plot(data['logarytmiczna_stopa_zwrotu'], label=ticker)
    axs[0].legend(loc='best')
    axs[0].set_title('Logarytmiczna stópa zwrotu', fontsize=9)

    """histogram,empiryczną funkcję gęstości,funkcję gęstości rozkładu normalnego z empiryczną średnią i wariancją"""
    ecdf = ECDF(data['logarytmiczna_stopa_zwrotu'])
    # axs[1].plot(ecdf.x, ecdf.y, label="empiryczna funkcja gęstości")
    hist, bins, _ = axs[1].hist(data['logarytmiczna_stopa_zwrotu'], bins=30, density=True,
             histtype='stepfilled', color='green',
             edgecolor='white')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    axs[1].plot(bin_centers, hist, label="empiryczna funkcja gęstości", color='red', linewidth=2)
    mu, std = norm.fit(data['logarytmiczna_stopa_zwrotu'])
    xmin, xmax = min(bin_centers), max(bin_centers)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[1].plot(x, p, 'k', linewidth=1, label='funkcja gęstośći dla rozkładu normalnego')
    axs[1].set_title('histogram,empiryczna funkcja gęstości,\nfunkcja gęstości dla rozkładu normalnego \nz'
                     ' empiryczną średnią i wariancją', fontsize=9)
    axs[1].legend(loc='best', fontsize=6)

    probplot(data['logarytmiczna_stopa_zwrotu'], plot=axs[2])
    nobs, minmax, mean, variance, skew, kurtosis = describe(data['logarytmiczna_stopa_zwrotu'])
    s_d = stdev(data['logarytmiczna_stopa_zwrotu'])
    info = f'średnia = {mean:>6.5f} \nodchylenie standardowe = {s_d:>6.5f} \nkurtoza = {kurtosis:>6.5f} \n' \
           f'skośność = {skew:>6.5f}'
    axs[2].text(x=.02, y=.75, s=info, transform=axs[2].transAxes, fontsize=7)
    axs[2].set_title('Wykers kwantyl-kwantyl', fontsize=9)
    fig.subplots_adjust(top=.9)
    fig.tight_layout()


def acf_pacf(ticker, data):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axs = plt.subplots(1, 2)

    plot_acf(data, lags=24, zero=True, ax=axs[0])

    plot_pacf(data, lags=24, zero=True, ax=axs[1])
    fig.subplots_adjust(top=.9)
