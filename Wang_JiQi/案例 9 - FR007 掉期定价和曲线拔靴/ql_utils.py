# -*- coding: utf-8 -*-
"""
@author: WangShengyuan
"""

import QuantLib as ql
import numpy as np
import re
import datetime
from convention_utils import curve_name


def ql2dtDate( qlDate ):
    try:
        len(qlDate)
        dtDate = [ datetime.datetime(d.year(), d.month(), d.dayOfMonth()) for d in qlDate ]
    except TypeError:
        dtDate = datetime.datetime(qlDate.year(), qlDate.month(), qlDate.dayOfMonth())
    return dtDate

def dt2qlDate( dtDate ):
    try:
        len(dtDate)
        qlDate = [ ql.Date(d.day, d.month, d.year) for d in dtDate ]
    except TypeError:
        qlDate = ql.Date(dtDate.day, dtDate.month, dtDate.year)
    return qlDate

def ql_dir( direction ):
    direc = direction.lower()[0]
    if direc == 'b':
        qldir = ql.DateGeneration.Backward
    elif direc == 'f':
        qldir = ql.DateGeneration.Forward
    return qldir

def ql_bdc( convention ):
    BDC = convention.lower().replace(' ','')
    if BDC in ('following', 'f'):
        qlbdc = ql.Following
    elif BDC in ('modifiedfollowing','mf'):
        qlbdc = ql.ModifiedFollowing
    elif BDC in ('preceding', 'p'):
        qlbdc = ql.Preceding
    elif BDC in ('modifiedpreceding','mp'):
        qlbdc = ql.ModifiedPreceding
    return qlbdc


def ql_period( ticker ):
    # ticker 可以是 3M, LIBOR 3M, LIBOR-3M, LIBOR_3M
    ticker = re.split('\s|_|-', curve_name(ticker))[-1]
    unit = ticker[-1]
    no_of_unit = int(ticker[:-1])
    
    unit_obj = ql.Years if unit == 'Y' else \
               ql.Months if unit == 'M' else \
               ql.Weeks if unit == 'W' else \
               ql.Days
    
    return ql.Period( no_of_unit, unit_obj )


def ql_FRA_period( ticker ):
    prd_list = re.split('X', ticker.upper())
    try: # 处理 axb，a 和 b 都是整数，比如 1x4 
        int(prd_list[0])
        prd_list = [p+'M' for p in prd_list]
    except ValueError: #处理 axb，a 和 b 都是字符串，比如 1Mx4M 
        pass
    
    return [ql_period(p) for p in prd_list]


def ql_freq( frequency ):
    f = frequency.lower()[0]
    return ql.Period(1, ql.Years) if f == 'a' else \
           ql.Period(6, ql.Months) if f == 's' else \
           ql.Period(3, ql.Months) if f == 'q' else \
           ql.Period(1, ql.Months) if f == 'm' else \
           ql.Period(2, ql.Weeks) if f == 'b' else \
           ql.Period(1, ql.Weeks) if f == 'w' else \
           ql.Period(1, ql.Days)


def ql_dc( dc ):
    dc = dc.lower().replace(' ','').replace('actual','act')
    
    if dc in ('act/act', 'act/actisda', 'act/actisma', 'act/acticma', 'act/actafb', 'a/a', 'a/aisda', 'a/aisma', 'a/aicma', 'a/aafb'):
        qldc = ql.ActualActual()
    elif dc in ('act/360', 'a/360'):
        qldc = ql.Actual360()
    elif dc in ('act/365', 'act/365F', 'a/365', 'a/365F'):
        qldc = ql.Actual365Fixed()
    elif dc in ('act/365(Canadian)', 'act/365F(Canadian)', 'a/365(Canadian)', 'a/365F(Canadian)'):
        qldc = ql.Actual365Fixed(ql.Actual365Fixed.Canadian)
    elif dc in ('act/365noleap', 'a/365noleap'):
        qldc = ql.Actual365NoLeap()
    elif dc in ('30/360', '30/360E'):
        qldc = ql.Thirty360()
    elif dc in ('act/252', 'a/252', 'bd/252', 'bus/252'):
        qldc = ql.Business252()
        
    return qldc