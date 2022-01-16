#! /anaconda3/bin/python3
# coding: utf-8
"""Docstring ici pour expliquer le module"""


import pandas as pd

def load_data(path=None):
    """docstring"""
    
    if path is None:
        rootpath = "./data/Flipkart/"
        path = rootpath + "flipkart_com-ecommerce_sample_1050.csv"
        data = pd.read_csv(path)
    
    else:
        data = pd.read_csv(path)

    return data


if __name__ == "__main__":
    load_data()
