# -*- coding: utf-8 -*-
"""
@author: WangShengyuan
"""

def cash_day_lag( curr ):
    curr = curr.upper()
    if curr in ('GBP', 'HKD', 'CNY'):
        day_lag = 0
    elif curr in ('KRW', 'AUD', 'CAD'):
        day_lag = 1
    else:
        day_lag = 2
    return day_lag

def swap_day_lag( curr ):
    curr = curr.upper()
    if curr in ('GBP', 'HKD'):
        day_lag = 0
    elif curr in ('CNY', 'KRW', 'AUD', 'CAD'):
        day_lag = 1
    else:
        day_lag = 2
    return day_lag

def FX_spot_convention( pair ):
    T_plus_1 = ('USDCAD','USDTRY','USDRUB','EURRUB')
    return 1 if pair.upper() in T_plus_1 else 2

def curve_name( curve_name ):
    curve_name = curve_name.upper()
    MAP = {'FR001':'FR 1D', 'FR007':'FR 7D', 'FR014':'FR 14D', 
           'FDR001':'FR 1D', 'FDR007':'FDR 7D', 'FDR014':'FDR 14D'}
    if curve_name in MAP.keys():
        return MAP[curve_name]
    else:
        return curve_name