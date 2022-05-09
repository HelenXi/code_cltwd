#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 08:45:22 2022

@author: xijiaqi
"""

import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from fractions import Fraction

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
n = 5000
ns = 1000
B = 800
replacement = [1,7/8,3/4,1/2]

swd = np.empty((ns,))
boot = np.empty((B,))

limit = gaussian_simulate(-2,2,1000,1000)
realden = gaussian_kde(limit)
me = np.mean(limit)
xs = np.linspace(me-5,me+10,1000)

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
densswd = gaussian_kde(swd,'silverman')



m = 1
for li in replacement:
    l = int(np.power(n,li))
    al, bl = np.ones((l,)) / l, np.ones((l,)) / l
    if l % 2 == 0:
        col = np.array([-2]*(l//2) + [2]*(l//2))
    else:
        col = np.array([-2]*(l//2+1)+[2]*(l//2))
    
    
        
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
    swd_boot = ot.emd2_1d(np.matmul(datap,direction.T),np.matmul(dataq,direction.T))
    
    for r in range(B):
        indices = np.random.choice(n,l,replace = True)
        rep = datap[indices]
        req = dataq[indices]
        for j in range(1000):
            theta = np.random.normal(0,1,size=(1,d))
            theta = theta/np.linalg.norm(theta)
            temp = ot.emd2_1d(np.matmul(rep,theta.T),col)
            if temp < val:
                val = temp
                direction = theta
        boot[r] = np.sqrt(l)*(ot.emd2_1d(np.matmul(rep,direction.T),np.matmul(req,direction.T)) - swd_boot)
    
    
    
    densboot = gaussian_kde(boot,'silverman')
    plt.figure(m)
    plt.plot(xs,densswd(xs),color='cadetblue')
    plt.fill_between(xs,densswd(xs),color='paleturquoise',alpha=0.3)
    plt.plot(xs,densboot(xs),color = 'darkolivegreen')
    plt.fill_between(xs,densboot(xs),color='palegreen',alpha = 0.4)
    plt.plot(xs,realden(xs),color='palevioletred')
    plt.fill_between(xs,realden(xs),color='pink',alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title('replacement l = n^'+str(Fraction(li)))
    m += 1