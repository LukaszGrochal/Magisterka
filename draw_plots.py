import matplotlib.pyplot as plt

def zwrot_logarytmiczny(ticker, data):
    plt.plot(data['logarytmiczna_stopa_zwrotu'], label=ticker)
    plt.legend(loc='best')


def zwrot_zwykly(ticker, data):
    plt.plot(data['zwykla_stopa_zwrotu'], label=ticker)
    plt.legend(loc='best')