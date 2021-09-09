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
            res[j] = (actual[j] - predicted[j]) / (actual[j] + predicted[j])
        return res

    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))


def compare_forecasts(y_true, y_pred):
    print(
        f'MSE={mse(y_true, y_pred)};'
        f' MAE={mae(y_true, y_pred)};'
        f' ME={mean_error(y_true, y_pred)};'
        f' TIC={tic(y_true, y_pred)};'
        f' AMAPE={amape(y_true, y_pred)}')
    return mse(y_true, y_pred), mae(y_true, y_pred), mean_error(y_true, y_pred), amape(y_true, y_pred), tic(y_true,
                                                                                                           y_pred)


# distributions = [Normal, GeneralizedError, StudentsT, SkewStudent]
distributions = ['normal', 'ged', 'studentst', 'skewstudent']
distributions_map = {'normal': 0, 'ged': 1, 'studentst': 2, 'skewstudent': 3}
means = ['zero', 'constant']
means_map = {'zero': 0, 'constant': 1}
pozycja = {'N.LLF': 0, 'AIC': 1, 'BIC': 2}

split_date = datetime.datetime(2021, 4, 1)

dane_nowe = data_from_2015_to_now()
tickers_models = dict()

for ticker in tickers:
    data = dane_nowe[ticker].logarytmiczna_stopa_zwrotu
    best_models = list()

    for vp in range(1, 5):
        for vq in range(5):
            for distribution in distributions:
                for srednia in means:
                    if ticker == 'TPE':
                        garch_model = arch_model(data, mean='AR', lags=1, dist=distribution, vol='GARCH', p=vp, q=vq, o=0)
                    else:
                        garch_model = arch_model(data, mean=srednia, dist=distribution, vol='GARCH', p=vp, q=vq, o=0)

                    gm_result = garch_model.fit(disp='off', last_obs=split_date, show_warning=False)


                    if len(best_models) > 2:
                        if (min(np.asarray(best_models)[:, pozycja['N.LLF']])) <= gm_result.loglikelihood:
                            if max(np.asarray(best_models)[:, pozycja['AIC']]) >= gm_result.aic or \
                                    max(np.asarray(best_models)[:, pozycja['BIC']]) >= gm_result.bic:
                                wynik = [int(gm_result.loglikelihood), int(gm_result.aic), int(gm_result.bic),
                                         means_map[srednia],
                                         distributions_map[distribution], vp, vq]

                                forecasts = gm_result.forecast(horizon=1, start=split_date, reindex=True)

                                wynik += [*compare_forecasts(data.loc[split_date:] ** 2,
                                                             forecasts.variance.loc[split_date:])]

                                best_models = sorted(sorted(best_models, key=lambda x: (np.asarray(x)[pozycja['AIC']]
                                                                                        ,
                                                                                        np.asarray(x)[pozycja['BIC']]))
                                                     , key=lambda x: np.asarray(x)[pozycja['N.LLF']], reverse=True)

                                best_models.pop(2)
                                wynik += [gm_result.params]
                                wynik += [gm_result.pvalues]
                                best_models.append(wynik)
                                best_models = sorted(sorted(best_models, key=lambda x: (np.asarray(x)[pozycja['AIC']]
                                                                                        ,
                                                                                        np.asarray(x)[pozycja['BIC']]))
                                                     , key=lambda x: np.asarray(x)[pozycja['N.LLF']], reverse=True)
                                print(best_models)

                    else:
                        wynik = [int(gm_result.loglikelihood), int(gm_result.aic), int(gm_result.bic),
                                 means_map[srednia],
                                 distributions_map[distribution], vp, vq]

                        forecasts = gm_result.forecast(horizon=1, start=split_date, reindex=True)

                        wynik += [*compare_forecasts(data.loc[split_date:] ** 2,
                                                     forecasts.variance.loc[split_date:])]
                        wynik += [gm_result.params]
                        wynik += [gm_result.pvalues]

                        best_models.append(wynik)

    tickers_models[ticker] = best_models

print(tickers_models)
print(f"\n\n\n\nTPE\n{tickers_models['TPE']}")
