# -*- coding: utf-8 -*-
"""
@author: WangShengyuan
"""

import QuantLib as ql
import re

CALENDAR_MAP = dict({
    'USNY': 'ql.UnitedStates()',
    'NY': 'ql.UnitedStates()',
    'USD': 'ql.UnitedStates()',
    
    'EUTA': 'ql.TARGET()', 
    'TARGET': 'ql.TARGET()', 
    'EUR': 'ql.TARGET()', 
    
    'GBLO': 'ql.UnitedKingdom()', 
    'LO': 'ql.UnitedKingdom()', 
    'GBP': 'ql.UnitedKingdom()', 
    
    'JPTO': 'ql.Japan()', 
    'TO': 'ql.Japan()',
    'JPY': 'ql.Japan()', 
        
    'AUSY': 'ql.Australia()', 
    'SY':'ql.Australia()',
    'AUD':'ql.Australia()', 
    
    'HKHK': 'ql.HongKong()', 
    'HK': 'ql.HongKong()', 
    'HKD': 'ql.HongKong()',
    
    'SGSG': 'ql.Singapore()', 
    'SG': 'ql.Singapore()', 
    'SGD': 'ql.Singapore()',
    
    'KRSE':'ql.SouthKorea()', 
    'SE':'ql.SouthKorea()', 
    'KRW':'ql.SouthKorea()', 
    
    'TWTA': 'ql.Taiwan()',
    'TA': 'ql.Taiwan()',
    'TWD':'ql.Taiwan()',
    
    'IDJA': 'ql.Indonesia()',
    'JA': 'ql.Indonesia()',
    'IDR':'ql.Indonesia()',
    
    'CNBJ': 'get_china_calendar()',
    'BJ': 'get_china_calendar()',
    'CNY': 'get_china_calendar()',
})


def get_calendar( code ):
    c = code.upper().replace(' ', '')
    return eval(CALENDAR_MAP[c])

def get_calendars( codes ):
    code_list = re.split('\s|_|-|; |, |\*|\n', codes)
    if len(code_list) == 1:
        return get_calendar(code_list[0])
    else:
        calendars = [ get_calendar(c) for c in code_list ]
        return ql.JointCalendar( *calendars )

def get_calendar_from_pair( pair ): 
    curr_str = pair[:3]+'-'+pair[-3:] 
    return get_calendars( curr_str )


def get_CDS_calendar( code ):
    if code.upper() in ('JPTO','JP','JPY'):
        cal = ql.NullCalendar()
        cal.addHoliday( ql.Date(20,3,2009) )
        cal.addHoliday( ql.Date(21,9,2009) )
        cal.addHoliday( ql.Date(22,9,2009) )
        cal.addHoliday( ql.Date(23,9,2009) )
        cal.addHoliday( ql.Date(22,3,2010) )
        cal.addHoliday( ql.Date(20,9,2010) )
        cal.addHoliday( ql.Date(21,3,2011) )
        cal.addHoliday( ql.Date(20,3,2012) )
        cal.addHoliday( ql.Date(20,3,2013) )
        cal.addHoliday( ql.Date(21,9,2015) )
        cal.addHoliday( ql.Date(22,9,2015) )
        cal.addHoliday( ql.Date(23,9,2015) )
        cal.addHoliday( ql.Date(21,3,2016) )
        cal.addHoliday( ql.Date(20,3,2017) )
        cal.addHoliday( ql.Date(21,9,2020) )
        cal.addHoliday( ql.Date(22,9,2020) )
        cal.addHoliday( ql.Date(23,9,2020) )
        cal.addHoliday( ql.Date(20,9,2021) )
        cal.addHoliday( ql.Date(21,3,2022) )
        cal.addHoliday( ql.Date(20,3,2024) )
        cal.addHoliday( ql.Date(20,3,2025) )
        cal.addHoliday( ql.Date(20,3,2026) )
        cal.addHoliday( ql.Date(21,9,2026) )
        cal.addHoliday( ql.Date(22,9,2026) )
        cal.addHoliday( ql.Date(23,9,2026) )
        cal.addHoliday( ql.Date(22,3,2027) )
        cal.addHoliday( ql.Date(20,9,2027) )
        cal.addHoliday( ql.Date(20,3,2028) )
        cal.addHoliday( ql.Date(20,3,2029) )
        cal.addHoliday( ql.Date(20,3,2030) )
        cal.addHoliday( ql.Date(20,9,2032) )
        cal.addHoliday( ql.Date(21,9,2032) )
        cal.addHoliday( ql.Date(22,9,2032) )
        cal.addHoliday( ql.Date(21,3,2033) )
        cal.addHoliday( ql.Date(20,3,2034) )
        cal.addHoliday( ql.Date(20,3,2036) )
        cal.addHoliday( ql.Date(22,9,2036) )
        cal.addHoliday( ql.Date(20,3,2037) )
        cal.addHoliday( ql.Date(21,9,2037) )
        cal.addHoliday( ql.Date(22,9,2037) )
        cal.addHoliday( ql.Date(23,9,2037) )
        cal.addHoliday( ql.Date(20,9,2038) )
        cal.addHoliday( ql.Date(21,3,2039) )
        cal.addHoliday( ql.Date(20,3,2040) )
        cal.addHoliday( ql.Date(20,3,2041) )
        cal.addHoliday( ql.Date(20,3,2042) )
        cal.addHoliday( ql.Date(21,9,2043) )
        cal.addHoliday( ql.Date(21,3,2044) )
        cal.addHoliday( ql.Date(20,3,2045) )
        cal.addHoliday( ql.Date(20,3,2046) )
        cal.addHoliday( ql.Date(20,3,2048) )
        cal.addHoliday( ql.Date(21,9,2048) )
        cal.addHoliday( ql.Date(22,9,2048) )
        cal.addHoliday( ql.Date(20,9,2049) )
        return ql.JointCalendar( cal, ql.WeekendsOnly() )
    else:
        return ql.WeekendsOnly()

    
def get_CI_calendar( index ):
    # CI = Credit Indice
    # CDX：CDX.NA.IG, CDX.HY.BBB, CDX.EM
    # iTraxx：iTraxx Europe, iTraxx Japan, iTraxx SovX CEEMEA
    
    token = re.split(r'[\.\s]', index) 
    
    if token[0].upper() == 'CDX':       # CDX
        return ql.UnitedStates()
    else:                               # iTraxx 
        if token[1].upper() == 'SOVX':  # iTraxx SovX
            return ql.UnitedStates()
        else:
            region = token[1].upper()
            if region == 'EUROPE':      # iTraxx Europe
                return ql.UnitedKingdom()
            elif region == 'JAPAN':     # iTraxx Japan
                return ql.Japan()
            elif region == 'AUSTRALIA': # iTraxx Australia
                return ql.Australia()
            elif region == 'ASIA':      # iTraxx Asia ex-Japan
                return ql.HongKong()
            else:                       # iTraxx CEEMEA
                return ql.UnitedStates()    

            
def get_china_calendar():
    cn_cal = ql.China()
    # 根据每年国务院发布部分节假日安排的通知来添加或删除中国节假日
    # 2019 - http://www.gov.cn/zhengce/content/2018-12/06/content_5346276.htm
    # 2020 - http://www.gov.cn/zhengce/content/2019-11/21/content_5454164.htm
    # 2021 - http://www.gov.cn/zhengce/content/2020-11/25/content_5564127.htm

    ########
    # 2019 #
    ########
    
    # 元旦：2018 年 12 月 30 日至 2019 年 1 月 1 日放假调休，共 3 天。2018 年 12 月 29 日（星期六）上班。
    cn_cal.removeHoliday( ql.Date(29,12,2018) )
    cn_cal.addHoliday( ql.Date(30,12,2018) )
    cn_cal.addHoliday( ql.Date(30,12,2018) )
    cn_cal.addHoliday( ql.Date(1,1,2019) )
    # 春节：2 月 4 日至 10 日放假调休，共 7 天。2 月 2 日（星期六）、2 月 3 日（星期日）上班。
    cn_cal.removeHoliday( ql.Date(2,2,2019) )
    cn_cal.removeHoliday( ql.Date(3,2,2019) )
    cn_cal.addHoliday( ql.Date(4,2,2019) )
    cn_cal.addHoliday( ql.Date(5,2,2019) )
    cn_cal.addHoliday( ql.Date(6,2,2019) )
    cn_cal.addHoliday( ql.Date(7,2,2019) )
    cn_cal.addHoliday( ql.Date(8,2,2019) )
    cn_cal.addHoliday( ql.Date(9,2,2019) )
    cn_cal.addHoliday( ql.Date(10,2,2019) )
    # 清明节：4 月 5 日放假，与周末连休。
    cn_cal.addHoliday( ql.Date(5,4,2019) )
    cn_cal.addHoliday( ql.Date(6,4,2019) )
    cn_cal.addHoliday( ql.Date(7,4,2019) )
    # 劳动节：5 月 1 日放假。
    cn_cal.addHoliday( ql.Date(1,5,2019) )
    # 端午节：6 月 7 日放假，与周末连休。
    cn_cal.addHoliday( ql.Date(7,6,2019) )
    cn_cal.addHoliday( ql.Date(8,6,2019) )
    cn_cal.addHoliday( ql.Date(9,6,2019) )    
    # 中秋节：9 月 13 日放假，与周末连休。
    cn_cal.addHoliday( ql.Date(13,9,2019) )
    cn_cal.addHoliday( ql.Date(14,9,2019) )
    cn_cal.addHoliday( ql.Date(15,9,2019) ) 
    # 国庆节：10 月 1 日至 7 日放假调休，共 7 天。9 月 29 日（星期日）、10 月 12 日（星期六）上班。
    cn_cal.removeHoliday( ql.Date(29,9,2019) )
    cn_cal.addHoliday( ql.Date(1,10,2019) )
    cn_cal.addHoliday( ql.Date(2,10,2019) )
    cn_cal.addHoliday( ql.Date(3,10,2019) )
    cn_cal.addHoliday( ql.Date(4,10,2019) )
    cn_cal.addHoliday( ql.Date(5,10,2019) )
    cn_cal.addHoliday( ql.Date(6,10,2019) )
    cn_cal.addHoliday( ql.Date(7,10,2019) )
    cn_cal.removeHoliday( ql.Date(12,10,2019) )
    
    ########
    # 2020 #
    ########
    
    # 元旦：2021 年 1 月 1 日放假，共 1 天。
    cn_cal.addHoliday( ql.Date(1,1,2020) )
    # 春节：1 月 24 日至 30 日放假调休，共 7 天。1 月 19 日（星期日）、2 月 1 日（星期六）上班。
    cn_cal.addHoliday( ql.Date(24,1,2020) )
    cn_cal.addHoliday( ql.Date(25,1,2020) )
    cn_cal.addHoliday( ql.Date(26,1,2020) )
    cn_cal.addHoliday( ql.Date(27,1,2020) )
    cn_cal.addHoliday( ql.Date(28,1,2020) )
    cn_cal.addHoliday( ql.Date(29,1,2020) )
    cn_cal.addHoliday( ql.Date(30,1,2020) ) 
    cn_cal.removeHoliday( ql.Date(19,1,2020) )
    cn_cal.removeHoliday( ql.Date(1,2,2020) )
    # 清明节：4 月 4 日至 6 日放假调休，共 3 天。
    cn_cal.addHoliday( ql.Date(4,4,2020) )
    cn_cal.addHoliday( ql.Date(5,4,2020) )
    cn_cal.addHoliday( ql.Date(6,4,2020) )
    # 劳动节：5 月 1 日至 5 日放假调休，共 5 天。4 月 26 日（星期日）、5 月 9 日（星期六）上班。
    cn_cal.removeHoliday( ql.Date(26,4,2020) )
    cn_cal.addHoliday( ql.Date(1,5,2020) )
    cn_cal.addHoliday( ql.Date(2,5,2020) )
    cn_cal.addHoliday( ql.Date(3,5,2020) )
    cn_cal.addHoliday( ql.Date(4,5,2020) )
    cn_cal.addHoliday( ql.Date(5,5,2020) )
    cn_cal.removeHoliday( ql.Date(9,5,2020) )
    # 端午节：6 月 25 日至 27 日放假，共 3 天。
    cn_cal.addHoliday( ql.Date(25,6,2020) )
    cn_cal.addHoliday( ql.Date(26,6,2020) )
    cn_cal.addHoliday( ql.Date(27,6,2020) )    
    # 国庆节、中秋节：10 月 1 日至 8 日放假调休，共 8 天。9 月 27 日（星期日）、10 月 10 日（星期六）上班。
    cn_cal.removeHoliday( ql.Date(27,9,2020) )
    cn_cal.addHoliday( ql.Date(1,10,2020) )
    cn_cal.addHoliday( ql.Date(2,10,2020) )
    cn_cal.addHoliday( ql.Date(3,10,2020) )
    cn_cal.addHoliday( ql.Date(4,10,2020) )
    cn_cal.addHoliday( ql.Date(5,10,2020) )
    cn_cal.addHoliday( ql.Date(6,10,2020) )
    cn_cal.addHoliday( ql.Date(7,10,2020) )
    cn_cal.addHoliday( ql.Date(8,10,2020) )
    cn_cal.removeHoliday( ql.Date(10,10,2020) )
    
    ########
    # 2021 #
    ########
    
    # 元旦：2021 年 1 月 1 日至 3 日放假，共 3 天。
    cn_cal.addHoliday( ql.Date(1,1,2021) )
    cn_cal.addHoliday( ql.Date(2,1,2021) )
    cn_cal.addHoliday( ql.Date(3,1,2021) )
    # 春节：2 月 11 日至 17 日放假调休，共 7 天。2 月 7 日（星期日）、2 月 20 日（星期六）上班。
    cn_cal.addHoliday( ql.Date(11,2,2021) )
    cn_cal.addHoliday( ql.Date(12,2,2021) )
    cn_cal.addHoliday( ql.Date(13,2,2021) )
    cn_cal.addHoliday( ql.Date(14,2,2021) )
    cn_cal.addHoliday( ql.Date(15,2,2021) )
    cn_cal.addHoliday( ql.Date(16,2,2021) )
    cn_cal.addHoliday( ql.Date(17,2,2021) )
    cn_cal.removeHoliday( ql.Date(7,2,2021) )
    cn_cal.removeHoliday( ql.Date(20,2,2021) )
    # 清明节：4 月 3 日至 5 日放假调休，共 3 天。
    cn_cal.addHoliday( ql.Date(3,4,2021) )
    cn_cal.addHoliday( ql.Date(4,4,2021) )
    cn_cal.addHoliday( ql.Date(5,4,2021) )
    # 劳动节：5 月 1 日至 5 日放假调休，共 5 天。4 月 25 日（星期日）、5 月 8 日（星期六）上班。
    cn_cal.removeHoliday( ql.Date(25,4,2021) )
    cn_cal.addHoliday( ql.Date(1,5,2021) )
    cn_cal.addHoliday( ql.Date(2,5,2021) )
    cn_cal.addHoliday( ql.Date(3,5,2021) )
    cn_cal.addHoliday( ql.Date(4,5,2021) )
    cn_cal.addHoliday( ql.Date(5,5,2021) )
    cn_cal.removeHoliday( ql.Date(8,5,2021) )
    # 端午节：6 月 12 日至 14 日放假，共 3 天。
    cn_cal.addHoliday( ql.Date(12,6,2021) )
    cn_cal.addHoliday( ql.Date(13,6,2021) )
    cn_cal.addHoliday( ql.Date(14,6,2021) )    
    # 中秋节：9 月 19 日至 21 日放假调休，共 3 天。9 月 18 日（星期六）上班。
    cn_cal.removeHoliday( ql.Date(18,9,2021) )
    cn_cal.addHoliday( ql.Date(19,9,2021) )
    cn_cal.addHoliday( ql.Date(20,9,2021) )
    cn_cal.addHoliday( ql.Date(21,9,2021) )
    # 国庆节：10 月 1 日至 7 日放假调休，共 7 天。9 月 26 日（星期日）、10 月 9 日（星期六）上班。
    cn_cal.removeHoliday( ql.Date(26,9,2021) )
    cn_cal.addHoliday( ql.Date(1,10,2021) )
    cn_cal.addHoliday( ql.Date(2,10,2021) )
    cn_cal.addHoliday( ql.Date(3,10,2021) )
    cn_cal.addHoliday( ql.Date(4,10,2021) )
    cn_cal.addHoliday( ql.Date(5,10,2021) )
    cn_cal.addHoliday( ql.Date(6,10,2021) )
    cn_cal.addHoliday( ql.Date(7,10,2021) )
    cn_cal.removeHoliday( ql.Date(9,10,2021) )
    
    return cn_cal