# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from market_variable import get_discount, get_forward_rate, update_fixing
from formatter import df_formatter, date_formatter
from utils import PoR
from convention_utils import curve_name
from daycount_utils import yearfrac
from date_utils import asdatetime, asdatetimes, asqldate, asqldates
from schedule import IRS_schedule, get_settle_date, trim_date, IBOR_date

from pyecharts.charts import Bar, Tab
from pyecharts.components import Table

from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType


class IR_InterestRateSwap():
    def __init__(self, date_eff, date_mat, N, curr, discCurve, 
                 leg1, f1, dc1, r, 
                 leg2, f2, dc2, refCurve, s, 
                 payCLD, fixingCLD, BDC):
        
        self.date_eff = asqldate(date_eff)
        self.date_mat = asqldate(date_mat)
        self.N = N
        self.curr = curr
        self.discCurve = discCurve
        self.leg1 = leg1
        self.f1 = f1
        self.dc1 = dc1
        self.r = r/100
        self.leg2 = leg2
        self.f2 = f2
        self.dc2 = dc2
        self.refCurve = refCurve
        self.s = s/100
        self.payCLD = payCLD
        self.fixingCLD = fixingCLD
        self.BDC = BDC
        self.generate_schedule()
        
    def generate_schedule(self):
        self.schedule = IRS_schedule( self.date_eff, self.date_mat, self.curr, self.refCurve, self.f1, self.f2, 
                                      self.payCLD, self.fixingCLD, bdc=self.BDC )

        
class IR_InterestRateSwap_pricer():
    
    def __init__( self, date_eval, market, deal, fixing ):
        self.date_eval = asqldate(date_eval)
        self.market = market
        self.deal = deal
        self.fixing = fixing
        self.__generate_date__()
    
    def __generate_date__(self):
        deal = self.deal
        self.date_settle = get_settle_date( self.date_eval, deal.curr )
        deal.schedule = {key:trim_date(self.date_settle, schedule) for key, schedule in deal.schedule.items() }
        self.deal = deal
    
    def __risk_factor__(self):
        market, deal, fixing = self.market, self.deal, self.fixing
        self.DISC_CURVE = market[deal.curr][curve_name(deal.discCurve)]
        self.REF_CURVE = market[deal.curr][curve_name(deal.refCurve)]
        self.FIXING = fixing[deal.curr][curve_name(deal.refCurve)]
    
    def __fixed_leg_pricer__(self):
        deal = self.deal
        schedule = deal.schedule['fixed leg schedule']
        date_start, date_end, date_pay = schedule['start_dates'], schedule['end_dates'], schedule['payment_dates']
        
        # 根据支付端或接收端得到 -1 或 1
        w = PoR(deal.leg1)
    
        # 根据日期计数惯例，计算每期的年限
        tau = np.array( [yearfrac(ds, de, deal.dc1) for ds, de in zip(date_start, date_end)] )
    
        # 计算每期现金流
        cash_flow = deal.N * tau * deal.r
    
        # 计算每期结束日上的折线因子
        DF = get_discount(self.DISC_CURVE, date_pay, self.date_eval)
    
        # 计算每期折线现金流
        disc_cash_flow = DF * cash_flow
    
        # 计算现值
        PV = w * np.sum(disc_cash_flow)
    
        # 打包所有中间产出
        self.output1 = {'Start Date': date_formatter(date_start), 'End Date': date_formatter(date_end), 
                        'Tenor': tau,  'Payment Date': date_formatter(date_pay),  'Discount Factor': DF, 
                        'Cash Flow': cash_flow, 'Discounted Cash Flow': disc_cash_flow,
                        'PorR':w, 'PV': PV}
    
    def __float_leg_pricer__(self):
        deal = self.deal
        schedule = deal.schedule['floating leg schedule']
        date_start, date_end, date_pay, IBOR_start_date, IBOR_end_date, IBOR_fixing_date = \
        schedule['start_dates'], schedule['end_dates'], schedule['payment_dates'], \
        schedule['index_start_dates'], schedule['index_end_dates'], schedule['index_fixing_dates']
        
        # 根据支付端或接收端得到 -1 或 1
        w = PoR(deal.leg2)

        # 根据日期计数惯例，计算每期的年限
        tau = np.array( [yearfrac(ds, de, deal.dc2) for ds, de in zip(date_start, date_end)] )

        # 计算每期的 IBOR
        IBOR = get_forward_rate( self.REF_CURVE, IBOR_start_date, IBOR_end_date, self.date_eval )

        # 更新定盘 IBOR
        IBOR = update_fixing( IBOR, IBOR_fixing_date, self.date_eval, self.FIXING )

        # 计算每期现金流
        cash_flow = deal.N * tau * (IBOR + deal.s)

        # 计算每期结束日上的折现因子
        DF = get_discount(self.DISC_CURVE, date_pay, self.date_eval)

        # 计算每期折线现金流
        disc_cash_flow = DF * cash_flow

        # 计算现值
        PV = w * np.sum(disc_cash_flow)

        # 打包所有中间产出
        self.output2 = {'Start Date': date_formatter(date_start), 'End Date': date_formatter(date_end), 
                        'Tenor': tau, 'IBOR': IBOR, 'Spread': deal.s, 
                        'Fixing Date': date_formatter(IBOR_fixing_date),
                        'IBOR Start Date': date_formatter(IBOR_start_date), 
                        'IBOR End Date': date_formatter(IBOR_end_date),
                        'Payment Date': date_formatter(date_pay),  'Discount Factor': DF, 
                        'Cash Flow': cash_flow, 'Discounted Cash Flow': disc_cash_flow,
                        'PorR':w, 'PV': PV}
    
    def NPV(self):
        self.__risk_factor__()
        self.__fixed_leg_pricer__()
        self.__float_leg_pricer__()
        fixed_leg_PV, floating_leg_PV = self.output1['PV'], self.output2['PV']
        NPV = fixed_leg_PV + floating_leg_PV
        output = {'Fixed Leg PV': fixed_leg_PV, 'Floating Leg PV': floating_leg_PV, 'NPV': NPV}
        return output
        
    def display_fixed_leg(self):
        formatter = df_formatter()
        FIX_LEG = self.output1
        non_display_columns = ['PorR','PV']
        df = pd.DataFrame(FIX_LEG).drop(columns=non_display_columns)
        return df.style.format({ k:formatter[k] for k in FIX_LEG.keys() if k not in non_display_columns} )\
                .background_gradient(subset=["Cash Flow", "Discounted Cash Flow"], cmap='PuBuGn')
        
    def display_floating_leg(self):
        formatter = df_formatter()
        FLT_LEG = self.output2
        non_display_columns = ['PorR','PV']
        df = pd.DataFrame(FLT_LEG).drop(columns=non_display_columns)
        df['IBOR'] = df['IBOR'].mul(100)
        return df.style.format({ k:formatter[k] for k in FLT_LEG.keys() if k not in non_display_columns} )\
                .background_gradient(subset=["IBOR", "Cash Flow", "Discounted Cash Flow"], cmap='PuBuGn')
    
    
    def __Table__( self, data ) -> Table:
        df = pd.DataFrame(data).drop(columns=['PorR','PV'])

        table = Table()

        headers = df.columns.tolist()
        rows = df.values.tolist()

        table.add( headers, rows )
        table.set_global_opts( title_opts = opts.ComponentTitleOpts(title= f"现值 = {data['PV']}"))

        return table
    
    def __Bar__( self, data1, data2 ) -> Bar:

        CF1 = data1['PorR']*data1['Cash Flow']
        CF2 = data2['PorR']*data2['Cash Flow']

        label_fmt = "function(params) {return params.data.toFixed(2)}"
        tooltip_fmt = "function(params) {return '日期：' + params.name + '<br>' + '数额：' + params.data.toFixed(2)}"

        bar = ( Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
               .add_xaxis(data1['Payment Date'])
               .add_yaxis("固定端", CF1.tolist(), category_gap="80%", label_opts=opts.LabelOpts(is_show=False))
               .add_xaxis(data2['Payment Date'])
               .add_yaxis("浮动端", CF2.tolist(), category_gap="80%", label_opts=opts.LabelOpts(is_show=False))
               .set_series_opts( label_opts=opts.LabelOpts( position="right", formatter=JsCode(label_fmt)) )
               .set_series_opts( tooltip_opts=opts.TooltipOpts( formatter=JsCode(tooltip_fmt)) )
               .set_global_opts(title_opts=opts.TitleOpts(title="现金流"),
                                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45, font_size=10)),
                                yaxis_opts=opts.AxisOpts(name=self.deal.curr),
                                datazoom_opts=[opts.DataZoomOpts(type_='inside')])
              )

        return bar
    
    def plot(self):
        FIX_LEG, FLT_LEG = self.output1, self.output2
        tab = Tab()
        tab.add( self.__Bar__( FIX_LEG, FLT_LEG ), "Cash Flow Chart")
        tab.add( self.__Table__(FIX_LEG), "Fixed Leg")
        tab.add( self.__Table__(FLT_LEG), "Floating Leg")
        return tab