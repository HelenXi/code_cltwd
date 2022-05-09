#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:56:50 2022

@author: xijiaqi
"""

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def gaussian_simulate(low,high,m,n):
    delta = (high-low)/m
    cov = np.ones((m,m))
    for i in range(m):
        for j in range(i,m):
            cov[i,j] =  cov[j,i] = 2*(low + delta*i)*(low+delta*j)
    data = np.empty((n,))
    for j in range(n):
        data[j] = np.max(np.random.multivariate_normal(np.zeros((m,)),cov))
    return data

limit = gaussian_simulate(-2,2,1000,500)
limDen = gaussian_kde(limit)
me = np.mean(limit)
xs = np.linspace(me-5,me+5,1000)
plt.plot(xs,limDen(xs))


    
    
    