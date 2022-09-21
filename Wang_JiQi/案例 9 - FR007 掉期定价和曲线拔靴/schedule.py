# -*- coding: utf-8 -*-

import QuantLib as ql
import datetime
import numpy as np
import pandas as pd

from convention_utils import *
from calendar_utils import *
from ql_utils import *

################################
# GENERIC FUNCTION
################################

def get_settle_date( trade_date:ql.Date, curr:str, day_lag=None ):
    day_lag = swap_day_lag(curr) if day_lag is None else day_lag
    cal = get_calendars( curr )
    settle_date = cal.advance( trade_date, ql.Period(day_lag, ql.Days) )
    return settle_date

def get_maturity_date( settle_date:ql.Date, cal, tickers:list ):
    return [cal.advance(date_settle, ql.Period(ticker)) for ticker in tickers]

def get_maturity_date_from_trade_date( trade_date:ql.Date, curr:str, tickers:list, day_lag=None ):
    settle_date = get_settle_date( trade_date, curr, day_lag )
    cal = get_calendars( curr )
    return [cal.advance(settle_date, ql.Period(ticker)) for ticker in tickers]

def trim_date( cutoff_date, schedule ):
    start_dates = schedule['start_dates']
    
    if cutoff_date < start_dates[0]:
        return schedule
    else:
        new_schedule = {}
        idx = sum([cutoff_date >= d for d in start_dates])
        for key, date_list in schedule.items():
            new_schedule[key] = date_list[idx-1:]
        return new_schedule

################################
# FOERIGN EXCHANGE CLASS
################################

def FX_value_date( pair:str, trade_date:ql.Date ):
    # EXAMPLE
    # FX_value_date( 'EURUSD', ql.Date(20, 11, 2018) )
    # FX_value_date( 'EURUSD', ql.Date(21, 11, 2018) )
    # FX_value_date( 'USDJPY', ql.Date(20, 11, 2018) )
    # FX_value_date( 'USDJPY', ql.Date(21, 11, 2018) )
    # FX_value_date( 'EURUSD', ql.Date(22, 11, 2018) )
    # FX_value_date( 'EURUSD', ql.Date(23, 11, 2018) )
    # FX_value_date( 'EURJPY', ql.Date(22, 11, 2018) )
    # FX_value_date( 'EURJPY', ql.Date(23, 11, 2018) )
    
    pair = pair.upper()
    nBD = FX_spot_convention( pair )
    curr1, curr2 = pair[:3], pair[-3:] 
    
    curr_str = curr1+'-'+curr2 
    
    if 'USD' == curr1:
        non_us_cal, all_cal = get_calendar(curr2), get_calendars(curr_str)
    elif 'USD' == curr2:
        non_us_cal, all_cal = get_calendar(curr1), get_calendars(curr_str)
    else: # USD is not in the pair (EURJPY, EURCNY)
        cur1_cal, cur2_cal, all_cal = get_calendar(curr1), get_calendar(curr2), get_calendars('USD-'+curr_str)
    
    us_cal = get_calendar('USD')
        
    if nBD == 1:
        value_date = all_cal.advance( trade_date, ql.Period(1,ql.Days) )
    else:
        if 'USD' in (curr1, curr2): # leading currency pair
            value_date = non_us_cal.advance( trade_date, ql.Period(2,ql.Days) )
        else: #cross-currency pair
            date1 = cur1_cal.advance( trade_date, ql.Period(2,ql.Days) )
            date2 = cur2_cal.advance( trade_date, ql.Period(2,ql.Days) )
            value_date = date1 if date1 > date2 else date2
        
        if us_cal.isHoliday(value_date):
            value_date = all_cal.advance( value_date, ql.Period(1,ql.Days) )
            
    return value_date


def FX_trade_date( pair:str, value_date:ql.Date ):
    # EXAMPLE
    # FX_trade_date( 'EURUSD', ql.Date(23, 11, 2018) )
    # FX_trade_date( 'USDJPY', ql.Date(26, 11, 2018) )
    # FX_trade_date( 'EURUSD', ql.Date(26, 11, 2018) )
    # FX_trade_date( 'EURUSD', ql.Date(27, 11, 2018) )
    # FX_trade_date( 'EURJPY', ql.Date(27, 11, 2018) )
    
    pair = pair.upper()
    nBD = FX_spot_convention( pair )
    curr1, curr2 = pair[:3], pair[-3:] 
    
    curr_str = curr1+'-'+curr2 
    
    if 'USD' == curr1:
        non_us_cal, all_cal = get_calendar(curr2), get_calendars(curr_str)
    elif 'USD' == curr2:
        non_us_cal, all_cal = get_calendar(curr1), get_calendars(curr_str)
    else: # USD is not in the pair (EURJPY, EURCNY)
        non_us_cal, all_cal = get_calendars(curr_str), get_calendars('USD-'+curr_str)
    
    weekend_cal = ql.WeekendsOnly()
    
    if nBD == 1:
        trade_date = all_cal.advance( value_date, ql.Period(-1,ql.Days) )
    else:
        trade_date = value_date
        for _ in range(10):
            trade_date = weekend_cal.advance( trade_date, ql.Period(-1,ql.Days) )
            VD = FX_value_date( pair, trade_date )
            if VD == value_date:
                break
    
    return trade_date


def FX_forward_schedule( pair:str, trade_date:ql.Date, ticker:str ):
    # EXAMPLE
    # pair, trade_date, ticker = 'EURUSD', ql.Date(20,11,2018), '1M'
    # pair, trade_date, ticker = 'USDJPY', ql.Date(20,11,2018), '6M'

    period = ql_period(ticker)
    # spot date = trade date + spot lag
    spot_date = FX_value_date( pair, trade_date )
    
    # delivery date = spot date + period
    cal = get_calendar_from_pair( pair )
    delivery_date = cal.advance(spot_date, period, ql.ModifiedFollowing, True)
    
    output = {'Pair': pair, 'Period': period, 'Trade Date': trade_date,
              'Spot Date': spot_date, 'Delivery Date': delivery_date}
    return output


def FX_forward_swappoint_schedule( pair:str, trade_date:ql.Date, ticker:str  ):
    # EXAMPLE
    # pair, trade_date = 'EURUSD', ql.Date(2,3,2020)
    # ticker = ['O/N', 'T/N', '1W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y', '3Y', '4Y', '5Y']
    
    # spot date = trade date + spot lag
    spot_date = FX_value_date( pair, trade_date )
    cal = get_calendar_from_pair( pair )

    schedule = {}
    
    for tik in ticker:
        if tik.upper() in ('ON', 'O/N', 'OVERNIGHT'):
            start_date = trade_date
            delivery_date =  cal.advance(start_date, ql.Period(1, ql.Days))
        elif tik.upper() in ('TN', 'T/N', 'TOM NEXT', 'TOMORROW NEXT'):
            start_date = cal.advance(trade_date, ql.Period(1, ql.Days))
            delivery_date =  cal.advance(start_date, ql.Period(1, ql.Days))
        else:
            period = ql_period(tik)
            # delivery date = spot date + period
            start_date = spot_date
            delivery_date = cal.advance(start_date, period, ql.ModifiedFollowing, True)
    
        schedule[tik] = {'Start Date': start_date, 'End Date': delivery_date}
        
    output = {'Pair': pair, 'Period': ticker, 'Trade Date': trade_date, 'Spot Date': spot_date, 'Schedule': schedule}   
    
    return output


def FX_option_schedule( pair:str, trade_date:ql.Date, ticker:list ):
    # EXAMPLE
    # pair, trade_date = 'EURUSD', ql.Date(20, 5, 2013)
    # ticker = ['O/N', '1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y', '2Y']

    spot_date = FX_value_date( pair, trade_date )
    cal = get_calendar_from_pair( pair )

    expiry_date_list = []
    delivery_date_list = []
    
    for tik in ticker:        
        if tik[-1].upper() in ('M', 'Y'): # monthly and yearly option

            period = ql_period(ticker)
            # delivery date = spot date + period
            cal = get_calendar_from_pair( pair )
            delivery_date = cal.advance(spot_date, period, ql.ModifiedFollowing, True)

            # expiry date = delivery date - spot lag
            expiry_date = FX_trade_date( pair, delivery_date )
            expiry_date_list.append(expiry_date)

        else: # daily and weekly option

            period = ql.Period(1, ql.Days) if ticker.upper() in ('ON','O/N','OVERNIGHT') else ql_period(ticker)

            # expiry date = trade date + period
            cal = ql.WeekendsOnly()
            expiry_date = cal.advance(trade_date, period) # expiry can be holiday but not weekend

            # delivery date = spot date + period
            delivery_date = FX_value_date( pair, expiry_date )
            delivery_date_list.append(delivery_date)
        
    output = {'Pair': pair, 'Period': ticker, 'Trade Date': trade_date, 'Spot Date': spot_date, 
              'Expiry Date': expiry_date, 'Delivery Date': delivery_date}   
    
    return output

################################
# INTEREST RATE CLASS
################################

def CD_schedule( trade_date:ql.Date, curr:str, index:str, period:str, pay_delay:int=0 ):
   
    calendar = get_calendar( curr )
        
    # CD dates
    d = cash_day_lag( curr )
    spot_date = calendar.advance( trade_date, ql.Period(d,ql.Days) ) 
    CD_period = ql_period( period )
    maturity_date = calendar.advance( spot_date, CD_period ) 
    pay_date = calendar.advance( maturity_date, ql.Period(pay_delay,ql.Days) )
    
    # IBOR dates
    IBOR_period = ql_period( index )
    index_start_date = spot_date
    index_end_date = calendar.advance( index_start_date, IBOR_period ) 
    
    output = dict( product_dates=dict(trade_date=trade_date,
                                      spot_date=spot_date,
                                      maturity_date=maturity_date,
                                      payment_date=pay_date),
                   index_dates=dict(index_fixing_date=trade_date,
                                    index_start_date=index_start_date,
                                    index_end_date=index_end_date) )
    return output


def FRA_schedule( trade_date:ql.Date, curr:str, index:str, period:str, pay_delay:int=0 ):
    # period = 'a⨯b' = '1⨯4','3⨯9','12⨯24', or
    #        = 'aM⨯bM' = '1M⨯4M','3M⨯9M','12M⨯24M'
    
    calendar = get_calendar( curr )
    
    # FRA dates          
    d = cash_day_lag( curr )
    spot_date = calendar.advance(trade_date, ql.Period(d,ql.Days)) 
    period1, period2 = ql_FRA_period( period )
    effective_date = calendar.advance( spot_date, period1 ) 
    maturity_date = calendar.advance( spot_date, period2 )
    pay_date = calendar.advance( maturity_date, ql.Period(pay_delay,ql.Days) )
    
    # IBOR dates
    IBOR_period = ql_period( index )
    index_start_date = effective_date
    index_end_date = calendar.advance( index_start_date, IBOR_period )
    fixing_date = calendar.advance( effective_date, ql.Period(-d, ql.Days) )
    
    output = dict( product_dates=dict(trade_date=trade_date,
                                      spot_date=spot_date,
                                      effective_date=effective_date,
                                      maturity_date=maturity_date,
                                      payment_date=pay_date),
                   index_dates=dict(index_fixing_date=fixing_date,
                                    index_start_date=index_start_date,
                                    index_end_date=index_end_date) )    
    
    return output


def IRF_schedule( trade_date:ql.Date, curr:str, index:str, code:str ):
    # code = 'EDH6', 'EPZ1'
    
    calendar = get_calendar( curr )
    
    # IRF dates                  
    d = cash_day_lag( curr )
    IMM_date = ql.IMM.date(code[-2:], trade_date)

    if curr == 'AUD': # SFE BBSW 3M: futures start date is the second Friday of the month
        IMM_date = IMM_date - 5 # from the 3rd Wedesday of month to the 2nd Friday of the month, simply minus 5

    effective_date = calendar.advance( IMM_date, ql.Period(1,ql.Days) ) if calendar.isHoliday(IMM_date) else IMM_date
    IBOR_period = ql_period( index )
    maturity_date = calendar.advance( effective_date, IBOR_period )
    
    if curr == 'AUD': # AUD BBSW 3M futures maturity can fall on weekend
        maturity_date = calendar.advance( effective_date, IBOR_period, ql.Unadjusted )
        if maturity_date.weekday() in [2,3,4,5,6]: # not falls on weekend, 1 - Sunday, 7 - Saturday
            maturity_date = calendar.adjust(maturity_date)
    
    # IBOR dates
    index_start_date = effective_date
    index_end_date = maturity_date
    fixing_date = calendar.advance( effective_date, ql.Period(-d, ql.Days) )
    
    output = dict( product_dates=dict(trade_date=trade_date,
                                      effective_date=effective_date,
                                      maturity_date=maturity_date),
                   index_dates=dict(index_fixing_date=fixing_date,
                                    index_start_date=index_start_date,
                                    index_end_date=index_end_date) )    
    
    return output


def date_series( effectiveDate, terminateDate, frequency, calendar,
                 convention='MF', end_date_convention='MF',direction='Forward', EOM=True, 
                 stubConvention=None, firstRegularDate=None, lastRegularDate=None):
    # stubConvention = None, 'both', 'short final','long final','smart final','short initial','long initial','smart initial'
    
    date_s = effectiveDate
    date_e = terminateDate
    freq = ql_freq(frequency)
    calendar = get_calendars(calendar)
    bdc = ql_bdc(convention)
    end_bdc = ql_bdc(end_date_convention)
    direc = ql_dir(direction)
    
    stub = stubConvention if stubConvention is None else stubConvention.lower() 
    
    if stub is None:
        schedule = ql.Schedule( date_s, date_e, freq, calendar, bdc, end_bdc, direc, EOM )
    else:
        if stub in ('short final', 'long final', 'smart final'):
            schedule = ql.Schedule( date_s, date_e, freq, calendar, bdc, end_bdc, ql_dir('forward'), EOM )
        elif stub in ('short initial', 'long initial', 'smart initial'):
            schedule = ql.Schedule( date_s, date_e, freq, calendar, bdc, end_bdc, ql_dir('backward'), EOM )
        elif stub == 'both':
            date_first = dt2qlDate( asdatetime(firstRegularDate) )
            date_last = dt2qlDate( asdatetime(lastRegularDate) )
            schedule = ql.Schedule( date_s, date_e, freq, calendar, bdc, end_bdc, direc, EOM, date_first, date_last )
    
    schedule_list = list(schedule)
    
    if stub == 'long final':
        schedule_list.pop(-2)
    elif stub == 'long initial':
        schedule_list.pop(1)
    elif stub == 'smart final': # equivalent of LONG_FINAL up to 7 days and SHORT_FINAL beyond that
        # The schedule periods will be determined forwards from the regular period start date.
        # Any remaining period, shorter than the standard frequency, will be allocated at the end.
        # If this results in a stub of less than 7 days, the stub will be combined with the next period.
        # If this results in a stub of 7 days or more, the stub will be retained.
        diff = schedule_list[-1] - schedule_list[-2]
        if diff < 7:
            schedule_list.pop(-2)
    elif stub == 'smart initial': # equivalent of LONG_INITIAL up to 7 days and SHORT_INITIAL beyond that. 
        # The schedule periods will be determined backwards from the regular period end date. 
        # Any remaining period, shorter than the standard frequency, will be allocated at the start. 
        # If this results in a stub of less than 7 days, the stub will be combined with the next period. 
        # If this results in a stub of 7 days or more, the stub will be retained. 
        diff = schedule_list[1] - schedule_list[0]
        if diff < 7:
            schedule_list.pop(1)
 
    output = dict(start_dates=schedule_list[:-1], end_dates=schedule_list[1:], all_dates=schedule_list )
    return output


def IBOR_date( start_date, curr, index, fixing_calendar, bdc='MF' ):
    
    cal = get_calendars(fixing_calendar)
    day_lag = swap_day_lag( curr )
    IBOR_tenor = ql_period(index)
    
    # IBOR start date, end date and fixing date
    IBOR_start_date = start_date
    IBOR_end_date = [ cal.advance(d, IBOR_tenor) for d in IBOR_start_date ]
    IBOR_fixing_date = [ cal.advance(d, ql.Period(-day_lag, ql.Days)) for d in IBOR_start_date ]
    
    output = dict(index_fixing_dates=IBOR_fixing_date, 
                  index_start_dates=IBOR_start_date, 
                  index_end_dates=IBOR_end_date)
    return output


def CMPR_date( start_date, end_date, curr, index, fixing_calendar, bdc='MF', direc='f' ):
    
    cal = get_calendars(fixing_calendar)
    day_lag = swap_day_lag(curr)
    CMPR_tenor = ql_period(index)
    
    period_start_date = []
    period_end_date = []
    CMPR_fixing_date = []
    CMPR_start_date = []
    CMPR_end_date = []

    for Ds, De in zip(start_date, end_date):
        schedule = ql.MakeSchedule( Ds, De, tenor=CMPR_tenor, rule=ql_dir(direc) )

        period_start, period_end = list(schedule)[:-1], list(schedule)[1:]
        
        CMPR_start = [ cal.adjust( d, ql_bdc(bdc)) for d in period_start ] 
        CMPR_end = [ cal.adjust(d+CMPR_tenor, ql_bdc(bdc)) for d in CMPR_start ]
        CMPR_fixing = [ cal.advance(d, ql.Period(-day_lag, ql.Days)) for d in CMPR_start ]
        
        period_start_date.append( period_start )
        period_end_date.append( period_end )
        CMPR_start_date.append( CMPR_start )
        CMPR_end_date.append( CMPR_end )
        CMPR_fixing_date.append( CMPR_fixing )
    
    output = dict(period_start_dates=period_start_date,
                  period_end_dates=period_end_date,
                  index_fixing_dates=CMPR_fixing_date, 
                  index_start_dates=CMPR_start_date, 
                  index_end_dates=CMPR_end_date)
    return output


def shift_date( dates, calendar, delay=0 ):
    if delay == 0:
        shift_dates = dates
    else:
        cal = get_calendars(calendar)
        shift_dates = [ cal.advance(d, ql.Period(delay,ql.Days)) for d in dates ]
    
    return shift_dates


def bond_schedule( date_eff, date_mat, freq, pay_cal, 
                   pay_delay=0, bdc='MF', end_date_bdc='MF',direc='Forward', EOM=False, stub=None, date_first=None, date_last=None ):
    
    schedule = date_series(date_eff, date_mat, freq, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    pay_dates = shift_date( schedule['end_dates'], pay_cal, pay_delay )
    schedule['payment_dates'] = pay_dates
    return schedule


def IRS_schedule( date_eff, date_mat, curr, index, fix_freq, flt_freq, pay_cal, fixing_cal, 
                  fix_pay_delay=0, flt_pay_delay=0, bdc='MF', end_date_bdc='MF',direc='Backward', EOM=True, 
                  stub=None, date_first=None, date_last=None):
 
    fix_schedule = date_series(date_eff, date_mat, fix_freq, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    flt_schedule = date_series(date_eff, date_mat, flt_freq, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    
    IBOR_schedule = IBOR_date( flt_schedule['start_dates'], curr, index, fixing_cal )
    fix_pay_dates = shift_date( fix_schedule['end_dates'], pay_cal, fix_pay_delay )
    flt_pay_dates = shift_date( flt_schedule['end_dates'], pay_cal, flt_pay_delay )
    
    fix_schedule['payment_dates'] = fix_pay_dates
    flt_schedule['payment_dates'] = flt_pay_dates
    
    flt_schedule = {**flt_schedule, **IBOR_schedule}
    
    output = {'fixed leg schedule': fix_schedule, 'floating leg schedule': flt_schedule}
    return output


def TS_schedule( date_eff, date_mat, curr, index1, index2, freq1, freq2, pay_cal, fixing_cal, 
                 pay_delay1=0, pay_delay2=0, bdc='MF', end_date_bdc='MF',direc='Forward', EOM=True, 
                 stub=None, date_first=None, date_last=None):
    
    schedule1 = date_series(date_eff, date_mat, freq1, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    schedule2 = date_series(date_eff, date_mat, freq2, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    
    IBOR_schedule1 = IBOR_date( schedule1['start_dates'], curr, index1, fixing_cal )
    IBOR_schedule2 = IBOR_date( schedule2['start_dates'], curr, index2, fixing_cal )
    
    pay_dates1 = shift_date( schedule1['end_dates'], pay_cal, pay_delay1 )
    pay_dates2 = shift_date( schedule2['end_dates'], pay_cal, pay_delay2 )
    
    schedule1['payment_dates'] = pay_dates1
    schedule2['payment_dates'] = pay_dates2
    
    schedule1 = {**schedule1, **IBOR_schedule1}
    schedule2 = {**schedule2, **IBOR_schedule2}
    
    output = {'leg 1 schedule': schedule1, 'leg 2 schedule': schedule2}
    return output


def CCBS_schedule( date_eff, date_mat, curr1, curr2, index1, index2, freq1, freq2, pay_cal, fixing_cal1, fixing_cal2,
                   pay_delay1=0, pay_delay2=0, bdc='MF', end_date_bdc='MF',direc='Forward', EOM=True, 
                   stub=None, date_first=None, date_last=None):
    
    schedule1 = date_series(date_eff, date_mat, freq1, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    schedule2 = date_series(date_eff, date_mat, freq2, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    
    IBOR_schedule1 = IBOR_date( schedule1['start_dates'], curr1, index1, fixing_cal1 )
    IBOR_schedule2 = IBOR_date( schedule2['start_dates'], curr2, index2, fixing_cal2 )
    
    pay_dates1 = shift_date( schedule1['end_dates'], pay_cal, pay_delay1 )
    pay_dates2 = shift_date( schedule2['end_dates'], pay_cal, pay_delay2 )
    
    schedule1['payment_dates'] = pay_dates1
    schedule2['payment_dates'] = pay_dates2
    
    schedule1 = {**schedule1, **IBOR_schedule1}
    schedule2 = {**schedule2, **IBOR_schedule2}
    
    output = {'leg 1 schedule': schedule1, 'leg 2 schedule': schedule2}
    return output


def CMPIRS_schedule( date_eff, date_mat, curr, index, fix_freq, flt_freq, pay_cal, fixing_cal, 
                     fix_pay_delay=0, flt_pay_delay=0, bdc='MF', end_date_bdc='MF',direc='Forward', EOM=True, 
                     stub=None, date_first=None, date_last=None):
    
    fix_schedule = date_series(date_eff, date_mat, fix_freq, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    flt_schedule = date_series(date_eff, date_mat, flt_freq, pay_cal, bdc, end_date_bdc, direc, EOM, stub, date_first, date_last)
    
    CMPR_schedule = CMPR_date( flt_schedule['start_dates'], flt_schedule['end_dates'], curr, index, fixing_cal )
    fix_pay_dates = shift_date( fix_schedule['end_dates'], pay_cal, fix_pay_delay )
    flt_pay_dates = shift_date( flt_schedule['end_dates'], pay_cal, flt_pay_delay )
    
    fix_schedule['payment_dates'] = fix_pay_dates
    flt_schedule['payment_dates'] = flt_pay_dates
    
    flt_schedule = {**flt_schedule, **CMPR_schedule}
    
    output = {'fixed leg schedule': fix_schedule, 'floating leg schedule': flt_schedule}
    return output


################################
# CREDIT CLASS
################################

def CDS_schedule( trade_date:ql.Date, start_date:ql.Date, end_date:ql.Date, calendar ):

    step_in_date = trade_date + 1
    settle_date = calendar.advance(trade_date, ql.Period(3,ql.Days))

    schedule = ql.Schedule( start_date, end_date, ql.Period(ql.Quarterly),
                            calendar, ql.Following, ql.Unadjusted,
                            ql.DateGeneration.CDS, False )

    # Accrual begin date is the latest CDS date prior to T+1 or on T+1
    is_prior_to_or_on_step_in_date = [step_in_date >=d for d in schedule]
    idx = sum(is_prior_to_or_on_step_in_date)

    # if all False <=> idx = 0 <=> new CDS, CDS dates are the whole schedule
    # otherwise <=> idx > 0 <=>  old CDS, the index of accrual begin date is idx-1
    CDS_dates = list(schedule)[idx-1:] if idx > 0 else list(schedule)

    accrual_start_dates = CDS_dates[:-1]
    accrual_end_dates = [ d - 1 for d in CDS_dates[1:-1] ] + [CDS_dates[-1]]
    payment_dates = CDS_dates[1:-1] + [calendar.adjust(CDS_dates[-1], ql.Following)]
    
    output = dict(trade_date=trade_date, step_in_date=step_in_date, settle_date=settle_date,
                  CDS_dates=CDS_dates, accrual_start_dates=accrual_start_dates, 
                  accrual_end_dates=accrual_end_dates, payment_dates=payment_dates)
    return output


def nonstandard_CDS_schedule( trade_date:ql.Date, start_date:ql.Date, end_date:ql.Date, calendar ):

    step_in_date = trade_date + 1
    settle_date = calendar.advance(trade_date, ql.Period(3,ql.Days))

    schedule = ql.Schedule( start_date, end_date, ql.Period(ql.Quarterly),
                            calendar, ql.Following, ql.Unadjusted,
                            ql.DateGeneration.Backward, False )

    # Accrual begin date is the latest CDS date prior to T+1 or on T+1
    is_prior_to_or_on_step_in_date = [step_in_date >=d for d in schedule]
    idx = sum(is_prior_to_or_on_step_in_date)

    # if all False <=> idx = 0 <=> new CDS, CDS dates are the whole schedule
    # otherwise <=> idx > 0 <=>  old CDS, the index of accrual begin date is idx-1
    CDS_dates = list(schedule)[idx-1:] if idx > 0 else list(schedule)

    accrual_start_dates = CDS_dates[:-1]
    accrual_end_dates = [ d - 1 for d in CDS_dates[1:-1] ] + [CDS_dates[-1]]
    payment_dates = CDS_dates[1:-1] + [calendar.adjust(CDS_dates[-1], ql.Following)]
    
    output = dict(trade_date=trade_date, step_in_date=step_in_date, settle_date=settle_date,
                  CDS_dates=CDS_dates, accrual_start_dates=accrual_start_dates, 
                  accrual_end_dates=accrual_end_dates, payment_dates=payment_dates)
    return output


################################
# EQUITY CLASS
################################


################################
# COMMODITY CLASS
################################