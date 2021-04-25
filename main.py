import pandas as pd
import numpy as np
from format_data import data_from_2007_to_2011, data_from_2015_to_now







if __name__ == '__main__':
    dane_stare = data_from_2007_to_2011()
    dane_nowe = data_from_2015_to_now()
    print(dane_nowe)