#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:05:23 2022

@author: xijiaqi
"""

import ot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from fractions import Fraction

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



ns = 200
swd = np.empty((ns,))

limit = np.empty((200,))
cov = np.ones((2500,2500))
for i in range(100):
    if i < 50:
        u = abs(abs(np.sin(np.pi*(i/100+1/4))) - abs(np.cos(np.pi*(i/100+1/4))))**2/2
    else:
        u = abs(abs(np.sin(np.pi*((i-50)/100+5/4))) - abs(np.cos(np.pi*((i-50)/100+5/4))))**2/2
    for j in range(100):
        if j < 50:
            v = abs(abs(np.sin(np.pi*(j/100+1/4))) - abs(np.cos(np.pi*(j/100+1/4))))**2/2
        else:
            v = abs(abs(np.sin(np.pi*((j-50)/100+5/4))) - abs(np.cos(np.pi*((j-50)/100+5/4))))**2/2
        for k in range(25):
            for l in range(25):
                cov[i*25+k,j*25+l] = 2*(-u+2*u*k/25)*(-v+2*v*l/25)
for j in range(200):
    data = np.random.multivariate_normal(np.zeros((2500,)),cov)
    temp = 0
    for k in range(100):
        temp += np.max(np.append(data[k*25:(k+1)*25],np.array([0])))/100
    limit[j] = temp
limit = 2*limit
realden = gaussian_kde(limit,'silverman')
me = np.mean(limit)
xs = np.linspace(me-1,me+2,1000)

n = 8000
B = 500
replacement = [1,7/8,3/4,1/2]

m = 1

n_seed = 5
swd = np.empty((200,))
boot = np.empty((4,500))
a, b = np.ones((n,)) / n, np.ones((n,)) / n

for i in range(200):
    smp = np.empty((n_seed,))      
    datap = generate_uniform_segment(n)
    dataq = generate_uniform_segment(n)
    for seed in range(n_seed):
        smp[seed] = ot.sliced.sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
    swd[i] = np.mean(smp)
swd = np.sqrt(n)*swd
swd_mean = np.mean(swd)
densswd = gaussian_kde(swd,'silverman')
xs = np.linspace(swd_mean-1,swd_mean+2,1000)
    
m = 1
for li in replacement:
    l = int(np.power(n,li))
    al, bl = np.ones((l,)) / l, np.ones((l,)) / l
    
    '''datap = generate_uniform_segment(n)
    dataq = generate_uniform_segment(n)
    for seed in range(n_seed):
            smp[seed] = ot.sliced.sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
    swd_boot = np.mean(smp)'''
    btp = np.empty((n_seed,))
    for j in range(B):
        indices = np.random.choice(n,l,replace = True)
        rep = datap[indices]
        req = dataq[indices]
        for seed in range(n_seed):
            btp[seed] = ot.sliced.sliced_wasserstein_distance(rep, req, al, bl, 1000, seed=seed)**2
        boot[m-1,j] = np.sqrt(l)*(np.mean(btp))
    densboot = gaussian_kde(boot[m-1,:],'silverman')
    plt.figure(m)
    plt.plot(xs,densswd(xs),color='cadetblue')
    plt.fill_between(xs,densswd(xs),color='paleturquoise',alpha=0.3)
    plt.plot(xs,densboot(xs),color = 'darkolivegreen')
    plt.fill_between(xs,densboot(xs),color='palegreen',alpha = 0.4)
    '''plt.plot(xs,realden(xs),color='palevioletred')
    plt.fill_between(xs,realden(xs),color='pink',alpha=0.3)'''
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title('replacement l = n^'+str(Fraction(li)))
    m += 1




