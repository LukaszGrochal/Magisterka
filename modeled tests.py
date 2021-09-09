from arch import arch_model
from arch.univariate import Normal, GeneralizedError, StudentsT, SkewStudent, ConstantMean, ZeroMean
from scipy.stats import gmean

from format_data import data_from_2007_to_2011, data_from_2015_to_now
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from format_data import tickers
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import acorr_ljungbox

def fix_window(data, model):
    _garch_model = model

    index = data.index
    start_loc = 0
    end_loc = np.where(index >= '2021-4-1')[0].min()
    forecasts = {}
    for i in range(40):
        res = _garch_model.fit(first_obs=start_loc + i, last_obs=i + end_loc, disp='off')
        temp = res.forecast(horizon=1).variance
        fcast = temp.iloc[i + end_loc - 1]
        forecasts[fcast.name] = fcast
    print(' Done!')
    variance_fixedwin = pd.DataFrame(forecasts).T
    return variance_fixedwin


def expand_window(data, model):
    _garch_model = model
    index = data.index
    start_loc = 0
    end_loc = np.where(index >= '2021-4-1')[0].min()
    forecasts = {}
    for i in range(40):
        res = _garch_model.fit(first_obs=start_loc, last_obs=i + end_loc, disp='off')
        temp = res.forecast(horizon=1).variance
        fcast = temp.iloc[i + end_loc - 1]
        forecasts[fcast.name] = fcast
    print(' Done!')
    variance_expandwin = pd.DataFrame(forecasts).T
    return variance_expandwin


def mean_error(y_true, y_pred):
    def percentage_error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            res[j] = (actual[j] - predicted[j])
        return res

    return np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred)))


def mse(y_true, y_pred):
    def percentage_error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            res[j] = (actual[j] - predicted[j]) ** 2
        return res

    return np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred)))


def mae(y_true, y_pred):
    def percentage_error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            res[j] = (actual[j] - predicted[j])
        return res

    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))


def tic(y_true, y_pred):
    def _error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            res[j] = (actual[j] - predicted[j])
        return res

    return np.sqrt(np.mean(_error(np.asarray(y_true), np.asarray(y_pred)) ** 2)) / \
           (np.sqrt(np.mean(np.asarray(y_true) ** 2))
            + np.sqrt(np.mean(np.asarray(y_pred) ** 2)))


def amape(y_true, y_pred):
    def percentage_error(actual, predicted):
        res = np.empty(actual.shape)
        for j in range(actual.shape[0]):
            # if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / (actual[j] + predicted[j])
            # else:
            #     res[j] = predicted[j] / np.mean(actual)
        return res

    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))


def compare_forecasts(y_true, y_pred):
    print(
        f'MSE={mse(y_true, y_pred):.4};'
        f' MAE={mae(y_true, y_pred):.4};'
        f' ME={mean_error(y_true, y_pred):.4};'
        f' TIC={tic(y_true, y_pred):.4};'
        f' AMAPE={amape(y_true, y_pred):.4}')
    return mse(y_true, y_pred), mae(y_true, y_pred), mean_error(y_true, y_pred), amape(y_true, y_pred), tic(y_true,
                                                                                                           y_pred)


distributions = ['normal', 'ged', 'studentst', 'skewstudent']
distributions_map = {'normal': 0, 'ged': 1, 'studentst': 2, 'skewstudent': 3}
means = ['zero', 'constant']
means_map = {'zero': 0, 'constant': 1}

pozycja = {'N.LLF': 0, 'AIC': 1, 'BIC': 2}

split_date = datetime.datetime(2021, 4, 1)

dane_nowe = data_from_2015_to_now()
tickers_models = dict()


def last_tests(ticker, srednia, rozklad, vp, vq, wer):
    data = dane_nowe[ticker].logarytmiczna_stopa_zwrotu

    garch_model = arch_model(data, mean=srednia, dist=rozklad, vol='GARCH', p=vp, q=vq,
                             o=0)
    gm_result = garch_model.fit(disp='off', last_obs=split_date, show_warning=False)
    dane_test = gm_result.std_resid.dropna()
    print(f'Test Ljung-Box na logarytmicznych stopach zwrotu dla {ticker}')
    print(f'ticker = wig \n', acorr_ljungbox(dane_test,
                                             lags=[5, 10, 20], return_df=True))
    print('\n')
    print(f'Test Ljung-Box na logarytmicznych stopach zwrotu **2 dla {ticker}')
    print(f'ticker = {ticker} \n', acorr_ljungbox(dane_test ** 2,
                                             lags=[5, 10, 20], return_df=True))
    print('\n')
    for x in [1, 2, 5, 10, 20]:
        wartosc, p_wartosc, _, _ = het_arch(dane_test, nlags=x)
        print(f"  dla opóźnienia {x}: wartość = {wartosc:.2} ({p_wartosc:.2E})")

    forecasts = gm_result.forecast(horizon=1, start=split_date, reindex=False)

    plt.plot(forecasts.variance ** 0.5, color='green', label='forecast')
    plt.plot(gm_result.conditional_volatility, color='red', label='garch')
    plt.plot(data, color='grey', label='Daily Log Returns', alpha=0.4)
    plt.legend(loc='best')

    plt.savefig(f'plots/{ticker}/GARCH({vp}-{vq})_wer{wer}.png', format='png', dpi=600)
    plt.clf()
    print(ticker)
    compare_forecasts(data.loc[split_date:] ** 2,
                      forecasts.variance.loc[split_date:])


######################  WIG

last_tests('WIG', 'zero', 'skewstudent', 2, 1, 1)
last_tests('WIG', 'zero', 'skewstudent', 1, 1, 2)



######################  WIG20
last_tests('WIG20', 'zero', 'studentst', 2, 2, 1)
last_tests('WIG20', 'zero', 'studentst', 2, 1, 2)


######################  PKN
last_tests('PKN', 'zero', 'studentst', 1, 1, 1 )
last_tests('PKN', 'zero', 'ged', 2, 1      , 2 )

######################  kgh
last_tests('KGH', 'zero', 'studentst', 1, 1  , 1   )
last_tests('KGH', 'zero', 'skewstudent', 1, 1, 2)

######################  MIL
last_tests('MIL', 'constant', 'studentst', 1, 4, 1)
last_tests('MIL', 'zero', 'studentst', 2, 4    , 2 )

######################  ING
last_tests('ING', 'zero', 'ged', 1, 3, 1)
last_tests('ING', 'zero', 'ged', 1, 2, 2)

######################  PGN
last_tests('PGN', 'zero', 'studentst', 1, 2, 1)
last_tests('PGN', 'zero', 'studentst', 1, 1, 2)

######################  MBK
last_tests('MBK', 'constant', 'studentst', 1, 2, 1)
last_tests('MBK', 'zero', 'studentst', 1, 2    , 2 )

######################  PKO
last_tests('PKO', 'zero', 'studentst', 1, 1    , 1 )
last_tests('PKO', 'constant', 'studentst', 1, 1, 2)

######################  BOS
last_tests('BOS', 'constant', 'studentst', 1, 2, 1)
last_tests('BOS', 'constant', 'studentst', 1, 1, 2)

def last_tests_tpe(  vp, vq, srednia='AR', rozklad='skewstudent', ticker='TPE'):
    data = dane_nowe[ticker].logarytmiczna_stopa_zwrotu

    garch_model = arch_model(data, lags=1, mean=srednia, dist=rozklad, vol='GARCH', p=vp, q=vq,
                             o=0)
    gm_result = garch_model.fit(disp='off', last_obs=split_date, show_warning=False)
    dane_test = gm_result.std_resid.dropna()
    print(f'Test Ljung-Box na logarytmicznych stopach zwrotu dla {ticker}')
    print(f'ticker = wig \n', acorr_ljungbox(dane_test,
                                             lags=[5, 10, 20], return_df=True))
    print('\n')
    print(f'Test Ljung-Box na logarytmicznych stopach zwrotu **2 dla {ticker}')
    print(f'ticker = {ticker} \n', acorr_ljungbox(dane_test ** 2,
                                             lags=[5, 10, 20], return_df=True))
    print('\n')
    for x in [1, 2, 5, 10, 20]:
        wartosc, p_wartosc, _, _ = het_arch(dane_test, nlags=x)
        print(f"  dla opóźnienia {x}: wartość = {wartosc:.2} ({p_wartosc:.2E})")

    forecasts = gm_result.forecast(horizon=1, start=split_date, reindex=False)

    plt.plot(forecasts.variance ** 0.5, color='green', label='forecast')
    plt.plot(gm_result.conditional_volatility, color='red', label='garch')
    plt.plot(data, color='grey', label='Daily Log Returns', alpha=0.4)
    plt.legend(loc='best')

    plt.savefig(f'plots/{ticker}/GARCH({vp}-{vq}).png', format='png', dpi=600)
    plt.clf()
    print(ticker)
    compare_forecasts(data.loc[split_date:] ** 2,
                      forecasts.variance.loc[split_date:])


last_tests_tpe(1, 3)
last_tests_tpe(1, 2)
