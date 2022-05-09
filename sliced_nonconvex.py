#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 22:15:45 2022

@author: xijiaqi
"""


import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as stats

def generate_uniform_segment(n):
    data = np.zeros((n,2))
    for j in range(n):
        data[j] = np.array([np.random.uniform(-1,1),np.random.choice([-1,1])])
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




limit = np.empty((200,))
cov = np.ones((3000,3000))
for i in range(100):
    if i < 50:
        u = 2*abs(abs(np.sin(np.pi*(i/100+1/4))) - abs(np.cos(np.pi*(i/100+1/4))))**2
    else:
        u = 2*abs(abs(np.sin(np.pi*((i-50)/100+5/4))) - abs(np.cos(np.pi*((i-50)/100+5/4))))**2
    for j in range(100):
        if j < 50:
            v = 2*abs(abs(np.sin(np.pi*(j/100+1/4))) - abs(np.cos(np.pi*(j/100+1/4))))**2
        else:
            v = 2*abs(abs(np.sin(np.pi*((j-50)/100+5/4))) - abs(np.cos(np.pi*((j-50)/100+5/4))))**2
        for k in range(30):
            for l in range(30):
                cov[i*30+k,j*30+l] = 2*(-u+2*u*k/30)*(-v+2*v*l/30)
for j in range(200):
    data = np.random.multivariate_normal(np.zeros((3000,)),cov)
    temp = 0
    for k in range(100):
        temp += np.max(data[k*30:(k+1)*30])/100
    limit[j] = temp
realden = gaussian_kde(limit,'silverman')
me = np.mean(limit)
xs = np.linspace(me-1,me+2,1000)


ns = 200
swd = np.empty((ns,))
n_list = [8000]
n_seed = 5
smp = np.empty((n_seed,))
m = 1
ks_list = []
for n in n_list:
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    for i in range(ns):
        datap = generate_uniform_segment(n)
        dataq = generate_uniform_segment(n)
        for seed in range(n_seed):
            smp[seed] = ot.sliced.sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
        swd[i] = np.mean(smp)
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