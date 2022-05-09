#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:32:19 2022

@author: xijiaqi
"""

import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as stats

def generate_uniform_sphere(d,n):
    data = np.zeros((n,d))
    for j in range(n):
        temp = np.random.normal(0,1,size=(1,d))
        data[j] = temp/np.linalg.norm(temp) + np.array([np.random.choice([-2,2])]*3)
    return data

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

d = 3
k = 1

n_list = [50,100,1000,5000]

ns = 500
swd = np.empty((ns,))
limit = gaussian_simulate(-2,2,1000,1000)
realden = gaussian_kde(limit)
me = np.mean(limit)
xs = np.linspace(me-5,me+10,1000)


n_seed = 5
m = 1
ks_list = []
for n in n_list:
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    co = np.array([-2]*(n//2) + [2]*(n//2))
    for i in range(ns):
        datap = generate_uniform_sphere(d, n)
        dataq = generate_uniform_sphere(d, n)
        direction = np.ones((1,d))/np.sqrt(d)
        val = 1
        for j in range(1000):
            theta = np.random.normal(0,1,size=(1,d))
            theta = theta/np.linalg.norm(theta)
            temp = ot.emd2_1d(np.matmul(datap,theta.T),co)
            if temp < val:
                val = temp
                direction = theta
        swd[i] = ot.emd2_1d(np.matmul(datap,direction.T),np.matmul(dataq,direction.T))
    swd = np.sqrt(n)*swd 
    swd_mean = np.mean(swd)
    density = gaussian_kde(swd,'silverman')


    ks_list.append(stats.ks_2samp(swd, limit))
    plt.figure(m)
    plt.plot(xs,density(xs),color='cadetblue')
    plt.fill_between(xs, density(xs),color='paleturquoise',alpha=0.5)
    plt.plot(xs,realden(xs),color='palevioletred')
    plt.fill_between(xs,realden(xs),color='pink',alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title('sample size n = ' + str(n))
    m+=1
