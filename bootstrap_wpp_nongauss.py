#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 00:35:36 2022

@author: xijiaqi
"""

import numpy as np
import ot
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from fractions import Fraction

def generate_uniform_sphere(d,n,R):
    data = np.zeros((n,d))
    for j in range(n):
        temp = np.random.normal(0,1,size=(1,d))
        data[j] = R*temp/np.linalg.norm(temp)
    return data

def gaussian_simulate(d,m,n):
    vectors = generate_uniform_sphere(d,m,1)
    cov = np.ones((m,m))
    for i in range(m):
        for j in range(i,m):
            u = vectors[i]
            v = vectors[j]
            cov[i,j] =  cov[j,i] = np.sum([u[i]**2*v[i]**2 for i in range(3)]) + \
                ((u[0]*v[1]+u[1]*v[0])**2 + (u[0]*v[2]+u[2]*v[0])**2 + (u[2]*v[1]+u[1]*v[2])**2 + \
                 2*(u[0]*u[1]*v[0]*v[1] + u[0]*u[2]*v[0]*v[2] + u[2]*u[1]*v[2]*v[1]))/3 - 5/9
    data = np.empty((n,))
    for j in range(n):
        data[j] = np.max(np.random.multivariate_normal(np.zeros((m,)),cov))
    return data


d = 3
Rp = 1
Rq = 2
rwpp = 1/3


limit = gaussian_simulate(d,1000,1000)
realden = gaussian_kde(limit)
me = np.mean(limit)
xs = np.linspace(me-2,me+3,1000)

m = 1

n_seed = 10
ns = 200
B = 500
n = 10000
replacement = [1,7/8,3/4,1/2]
a, b = np.ones((n,)) / n, np.ones((n,)) / n
swd = np.empty((ns,))
boot = np.empty((B,))

for i in range(ns):
        smp = np.empty((n_seed,))
        datap = generate_uniform_sphere(d,n,Rp)
        dataq = generate_uniform_sphere(d,n,Rq)       
        for seed in range(n_seed):
            smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
        swd[i] = np.mean(smp)
swd = np.sqrt(n)*(swd - rwpp)
densswd = gaussian_kde(swd,'silverman')

m = 1
for li in replacement:
    l = int(np.power(n,li))
    al, bl = np.ones((l,)) / l, np.ones((l,)) / l    
    smp = np.empty((n_seed,))    
    datap = generate_uniform_sphere(d, n, Rp)
    dataq = generate_uniform_sphere(d, n, Rq)
    for seed in range(n_seed):
        smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
    swd_boot = np.mean(smp)
    for r in range(B):
        indices = np.random.choice(n,l,replace = True)
        rep = datap[indices]
        req = dataq[indices]
        for seed in range(n_seed):
            smp[seed] = ot.sliced.max_sliced_wasserstein_distance(rep, req, al, bl, 1000, seed=seed)**2
        boot[r] = np.sqrt(l)*(np.mean(smp) - swd_boot)    
    
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
