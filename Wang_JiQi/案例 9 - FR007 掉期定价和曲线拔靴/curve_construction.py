# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:35:19 2021

@author: WangShengyuan
"""

import numpy as np
import pandas as pd
import scipy.interpolate as spi
from scipy.optimize import fsolve
from date_utils import asdatetime, asqldate, asdatetimes, asqldates
from daycount_utils import yearfrac
from calendar_utils import get_calendar, get_calendars
from schedule import date_series, get_settle_date, get_maturity_date_from_trade_date


def bootstrapping( MARKET, curr, index, date_eval ):
    # 预处理从中国外汇交易中心下载的数据
    market_quote = process_data( MARKET, curr, index, date_eval )
    
    # 曲线上的标准年限：(到期日 - 估值日) / 365
    date_mat = [ asqldate(d) for d in market_quote['Date'] ]
    std_T = np.array( [ yearfrac(date_eval, d, 'Act/365') for d in date_mat ] )
    
    (fix_freq, pay_cal) = ('Quarterly', 'CNBJ')

    date_settle = get_settle_date( asqldate(date_eval), curr )
        
    r_vec = np.zeros(std_T.shape)

    for i, (ticker, inst) in enumerate(market_quote.iterrows()):

        # 1. 根据起息日、到日期、支付频率和一些关系生成日期表
        i_schedule = date_series( date_settle, date_mat[i], fix_freq, curr )

        # 2. 计算日期表到估值日的年限，用惯例 Act/365，用于插值
        i_T = np.array( [ yearfrac(date_eval, d, 'Act/365') for d in i_schedule['all_dates'] ] )

        # 3. 计算每期之间的年限，根据 CNY IRS 固定端决定，日期计数惯例 Act/365
        i_tau = np.array( [ yearfrac(ds, de, 'Act/365') for ds, de 
                            in zip(i_schedule['start_dates'], i_schedule['end_dates']) ] )

        # 4. Bootstrap 出零息利率
        K = inst['Quote'] / 100                                                 # 固定端利率
        obj_func = lambda x: compute_NPV( i, r_vec, std_T, K, i_tau, i_T, x )
        r0 = K if i==0 else r_vec[i-1]

        r = fsolve( obj_func, r0 )
        r_vec[i] = r
    
    curve = market_quote.copy()
    curve['Rate'] = r_vec*100
    curve['Discount'] = np.exp(-r_vec*std_T)
    curve = curve.set_index('Date')
    return curve


def compute_NPV( i, std_r, std_T, K, i_tau, i_T, r ):
    # 最后一个参数 r 是未知量，需要求解
    
    if i == 0: # 第一个标准日期的零息利率需要平外插 (flat extrapolation)
        
        TT = np.array( [0, std_T[i]] )
        rr = np.array( [r, r] )
        
    else: # 之后的零息利率线性内插 (linear extrapolation)
     
        TT = np.r_[0, std_T[:i], std_T[i]]        
        rr = np.r_[std_r[0], std_r[:i], r]
    
    
    # 创建样条对象 (spline object)   
    spline = spi.make_interp_spline( TT, rr, k=1 )
        
    i_r = np.squeeze( spline(i_T) )
    i_DF = np.exp(-i_r*i_T)
    
    fix_leg_PV = K * np.sum(i_tau*i_DF[1:])  
    floating_leg_PV = i_DF[0] - i_DF[-1]
    
    NPV = fix_leg_PV - floating_leg_PV
    
    return NPV


def process_data( MARKET, curr, index, date_eval ):
    
    try:
        data = MARKET[curr][index][date_eval.strftime('%Y-%m-%d')]
    except:
        data = MARKET[curr][index][date_eval]
    
    data = data.drop(columns=['曲线名称', '时刻','价格类型'])
    df = pd.DataFrame( index=data.columns )
    df.index.name= 'Period'
        
    date_mat = get_maturity_date_from_trade_date( asqldate(date_eval), curr, df.index )
    
    df['Date'] = asdatetimes(date_mat)
    df['Quote'] = data.values[0]
    return df