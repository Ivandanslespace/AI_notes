# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 00:24:58 2021

@author: WangShengyuan
"""

# Python 内置工具
import numpy as np
import pandas as pd

from pyecharts.charts import Bar, Tab
from pyecharts.components import Table

from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType

# 其他函数和类
from market_variable import get_discount, get_forward_rate, update_fixing
from formatter import df_formatter, date_formatter
from utils import PoR
from daycount_utils import yearfrac
from date_utils import asdatetime, asdatetimes, asqldate, asqldates
from schedule import CMPIRS_schedule, get_settle_date, trim_date, CMPR_date


from IR_InterestRateSwap_engine import IR_InterestRateSwap, IR_InterestRateSwap_pricer

#####################

class IR_CompoundIRS( IR_InterestRateSwap ):
    def __init__(self, date_eff, date_mat, N, curr, discCurve, 
                 leg1, f1, dc1, r, 
                 leg2, f2, dc2, refCurve, s, 
                 payCLD, fixingCLD, BDC):
        super().__init__(date_eff, date_mat, N, curr, discCurve, 
                 leg1, f1, dc1, r, 
                 leg2, f2, dc2, refCurve, s, 
                 payCLD, fixingCLD, BDC)
    
    def generate_schedule(self):
        self.schedule = CMPIRS_schedule( self.date_eff, self.date_mat, self.curr, self.refCurve, self.f1, self.f2, 
                                         self.payCLD, self.fixingCLD, bdc=self.BDC )

        
class IR_CompoundIRS_pricer( IR_InterestRateSwap_pricer ):
        
    def __init__( self, date_eval, market, deal, fixing ):
        super().__init__(date_eval, market, deal, fixing)
        
    def __float_leg_pricer__(self):
        deal = self.deal
        schedule = deal.schedule['floating leg schedule']
        date_start, date_end, date_pay, period_start_date, period_end_date, \
        index_start_date, index_end_date, index_fixing_date = \
        schedule['start_dates'], schedule['end_dates'], schedule['payment_dates'], \
        schedule['period_start_dates'], schedule['period_end_dates'], \
        schedule['index_start_dates'], schedule['index_end_dates'], schedule['index_fixing_dates']
                
        # 根据支付端或接收端得到 -1 或 1
        w = PoR(deal.leg2)
        
        # 根据日期计数惯例，计算每期的年限
        tau = np.array( [yearfrac(ds, de, deal.dc2) for ds, de in zip(date_start, date_end)] )

        # 计算每期的复合利率 CR
        FR007 = []
        CR = []
        CR_tau = []
        
        for i, (cmp_start, cmp_end, CR_start, CR_end, CR_fixing) in enumerate(zip(period_start_date,
                                                                                  period_end_date,
                                                                                  index_start_date,
                                                                                  index_end_date,
                                                                                  index_fixing_date)):
            
            R = get_forward_rate( self.REF_CURVE, CR_start, CR_end, self.date_eval, deal.dc2 )
                        
            if i == 0: # 更新定盘
                R = update_fixing( R, CR_fixing, self.date_eval, self.FIXING )
            
            cmp_tau = np.array( [yearfrac(ds, de, deal.dc2) for ds, de in zip(cmp_start, cmp_end)] )
            
            FR007.append(R)
            CR_tau.append(cmp_tau)
            CR.append( np.prod(1+R*cmp_tau) - 1 )

        CR = np.array(CR)/tau
        
        # 计算每期现金流
        cash_flow = deal.N * tau * (CR + deal.s)

        # 计算每期结束日上的折线因子
        DF = get_discount( self.DISC_CURVE, date_pay, self.date_eval )

        # 计算每期折线现金流
        disc_cash_flow = DF * cash_flow

        # 计算现值
        PV = w * np.sum(disc_cash_flow)

        # 打包所有中间产出
        self.output2 = {'Start Date': date_formatter(date_start), 'End Date': date_formatter(date_end), 
                        'Tenor': tau, 'Compound Rate': CR, 'Spread': deal.s, 
                        'Period Start Date': [date_formatter(d) for d in period_start_date],
                        'Period End Date': [date_formatter(d) for d in period_end_date],
                        'CR Fixing Date': [date_formatter(d) for d in index_fixing_date],
                        'CR Start Date': [date_formatter(d) for d in index_start_date], 
                        'CR End Date': [date_formatter(d) for d in index_end_date],
                        'FR007': FR007,
                        'FR007 Tenor': CR_tau,
                        'Payment Date': date_formatter(date_pay),  'Discount Factor': DF, 
                        'Cash Flow': cash_flow, 'Discounted Cash Flow': disc_cash_flow,
                        'PorR':w, 'PV': PV}


    def display_floating_leg(self):
        formatter = df_formatter()
        FLT_LEG = self.output2
        non_display_columns = ['PorR','PV','Period Start Date','Period End Date',\
                               'CR Fixing Date','CR Start Date','CR End Date','FR007','FR007 Tenor']
        df = pd.DataFrame(FLT_LEG).drop(columns=non_display_columns)
        
        df['Compound Rate'] = df['Compound Rate'].mul(100)
        return df.style.format({ k:formatter[k] for k in FLT_LEG.keys() if k not in non_display_columns} )\
                .background_gradient(subset=["Compound Rate", "Cash Flow", "Discounted Cash Flow"], cmap='PuBuGn')
    
    
    def __Table_fix__( self, data ) -> Table:
        df = pd.DataFrame(data).drop(columns=['PorR','PV'])

        table = Table()

        headers = df.columns.tolist()
        rows = df.values.tolist()

        table.add( headers, rows )
        table.set_global_opts( title_opts = opts.ComponentTitleOpts(title= f"现值 = {data['PV']}"))

        return table
  

    def __Table_flt__( self, data ):
        non_display_columns = ['PorR','PV','Period Start Date','Period End Date',\
                               'CR Fixing Date','CR Start Date','CR End Date','FR007','FR007 Tenor']
        df = pd.DataFrame(data).drop(columns=non_display_columns)
        df[['Compound Rate', 'Spread']] = df[['Compound Rate', 'Spread']].mul(100)
                
        table = Table()
        headers, rows = df.columns.tolist(), df.values.tolist()
        table.add( headers, rows )
        table.set_global_opts( title_opts = opts.ComponentTitleOpts(title= f"现值 = {data['PV']}"))
        
        df1 = pd.DataFrame(data)[['Period Start Date','Period End Date',\
                                  'CR Fixing Date','CR Start Date','CR End Date','FR007','FR007 Tenor']]
        table_list = []
        
        for i, s in df1.iterrows():
            d = {key:val for key, val in zip(s.index, s.values)}
            df_i = pd.DataFrame(d)
            df_i.index = np.arange(1,len(df_i)+1)
            df_i['FR007'] = df_i['FR007'].mul(100)
            headers, rows = df_i.columns.tolist(), df_i.values.tolist()
            table_i = Table()
            table_i.add( headers, rows )
            table_i.set_global_opts( title_opts = opts.ComponentTitleOpts(title= f"第 {i+1} 期的复合利率 (%) = {data['Compound Rate'][i]*100}"))
            table_list.append( table_i )
        
        return table, table_list
    
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
        tab.add( self.__Bar__( FIX_LEG, FLT_LEG ), "现金流图")
        tab.add( self.__Table_fix__(FIX_LEG), "固定端")
        (table, table_list) = self.__Table_flt__(FLT_LEG)
        tab.add( table, "浮动端")
        for i, t in enumerate(table_list,start=1):
            tab.add( t, f"第 {i} 期")
        return tab