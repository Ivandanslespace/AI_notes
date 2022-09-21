# -*- coding: utf-8 -*-

import datetime
from date_utils import asdatetime

def date_formatter( date, fmt='%Y/%m/%d' ):
    return [asdatetime(d).strftime(fmt) for d in date]


def df_formatter():
    money_fmt = "${:10,.2f}"
    pct_fmt = "{:.4f}%"
    date_fmt = lambda x: x.strftime('%Y/%m/%d') if not isinstance(x,str) else x
    num_fmt = "{:,.4f}"
    
    formatter = { "Cash Flow": money_fmt, 
                  "Discounted Cash Flow": money_fmt,
                  "Start Date": date_fmt,
                  "End Date": date_fmt,
                  "Fixing Date": date_fmt,
                  "Payment Date": date_fmt,
                  "IBOR Start Date": date_fmt,
                  "IBOR End Date": date_fmt,
                  "CMPR Start Date": date_fmt,
                  "CMPR End Date": date_fmt,
                  "IBOR": pct_fmt,
                  "Compound Rate": pct_fmt,
                  "Spread": pct_fmt,
                  "Tenor": num_fmt, 
                  "Discount Factor": num_fmt
                }
    return formatter
