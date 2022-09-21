# -*- coding: utf-8 -*-

import datetime
import numpy as np
import pandas as pd
import scipy.interpolate as spi

import QuantLib as ql
from date_utils import asdatetime, asdatetimes
from daycount_utils import yearfrac


def get_discount( curve, date, today ):
    today = asdatetime(today)
    curve_date = pd.DatetimeIndex(curve.index)
    r_std = curve['Rate']
    d_std = (curve_date - today).days.values
    
    if isinstance(date, (list, np.ndarray)):
        date = asdatetimes(date)
        d = np.array( [(dd-today).days for dd in date] )
    else:
        date = asdatetime(date)
        d = (date - today).days
        
    f = spi.interp1d(d_std, r_std, fill_value=(r_std[0], r_std[-1]), bounds_error=False)
    r = f(d)
    DF = np.exp(-d/365 * r/100)
    return DF


def get_forward_rate( curve, date_s, date_e, date_obs, dcc='act/360' ):
    if isinstance(date_s, (list, np.ndarray)):
        tau = np.array( [yearfrac(ds, de, dcc) for ds, de in zip(date_s, date_e)] )
    else:
        tau = yearfrac(date_s, date_e, dcc)
        
    DF_s = get_discount( curve, date_s, date_obs )
    DF_e = get_discount( curve, date_e, date_obs )
    F = (1/tau)*(DF_s/DF_e - 1)
    return F


def update_fixing( R, fixing_date, date_eval, fixing ):
    date_eval = asdatetime(date_eval)
    fixing_date = asdatetimes(fixing_date)
        
    new_R = R.copy()
    
    for i, d in enumerate(fixing_date):
        if date_eval >= d:
            new_R[i] = fixing.loc[d,:] / 100
        else:
            break
    return new_R