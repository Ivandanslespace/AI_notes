# -*- coding: utf-8 -*-

# import numpy as np
# import pandas as pd
# import scipy.interpolate as spi
# import datetime
# from daycount_utils import yearfrac


def PoR( leg ):
    Rec = ('r','rec','receive','receiving','receiver','b','buy','buying','buyer','loan')
    return 1 if leg.lower() in Rec else -1

def get_freq(f):
    if isinstance(f, str):
        n_of_period_per_year = {'a':1, 's':2, 'q':4, 'm':12}
        n_of_month_per_period = {'a':12, 's':6, 'q':3, 'm':1}
        output = ( n_of_period_per_year[f[0].lower()], n_of_month_per_period[f[0].lower()] )
    else:
        output = (int(f), int(12/f))
    return output

def get_day_lag( curr ):
    curr = curr.upper()
    if curr in ('GBP', 'AUD', 'HKD'):
        day_lag = 0
    elif curr in ('KRW', 'CAD', 'CNY'):
        day_lag = 1
    else:
        day_lag = 2
    return day_lag