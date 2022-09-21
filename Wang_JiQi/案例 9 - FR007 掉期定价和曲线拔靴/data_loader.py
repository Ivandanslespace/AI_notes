# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:38:31 2021

@author: WangShengyuan
"""
import os
import pandas as pd

def load_data():
    directory = os.getcwd() 

    INSTRUMENT = read_instrument( directory + '\\INSTRUMENT', 'csv' )
    MARKET = read_market( directory + '\\MARKET', 'xlsx', '日期' )
    FIXING = read_fixing( directory + '\\FIXING', 'xlsx', 'Date' )
    
    return INSTRUMENT, MARKET, FIXING


def read_instrument( path, file_type ):    
    obj_dict = {}
    file_type = file_type.lower()
        
    for r, d, f in os.walk(path):
        for file in f:
            if file_type in ['xls','xlsx']:
                if file_type in file:
                    df = pd.read_excel( os.path.join(r, file), sheet_name=None, index_col=0 )
            else:
                if file_type in file:
                    df = pd.read_csv( os.path.join(r, file), index_col=0 )
            
            key = file.split('.')[0]
            obj_dict[key] = df
            
    return obj_dict


def read_market( path, file_type, col ):    
    obj_dict = {}
    file_type = file_type.lower()
        
    for r, d, f in os.walk(path):
        for file in f:
            if file_type in ['xls','xlsx']:
                if file_type in file:
                    df = pd.read_excel( os.path.join(r, file), sheet_name=None, parse_dates=[col], index_col=col )
                    for k, v in df.items():
                        v = v.dropna(axis=0)
                        v.index = pd.to_datetime(v.index)
                        df[k] = v
            else:
                if file_type in file:
                    df = pd.read_csv( os.path.join(r, file), parse_dates=[col], index_col=col )
                    df = df.dropna(axis=0)
                    df.index = pd.to_datetime(df.index)
            
            key = file.split('.')[0]
            obj_dict[key] = df
            
    return obj_dict


def read_fixing( path, file_type, col ):    
    return read_market( path, file_type, col )