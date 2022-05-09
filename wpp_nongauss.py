#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 10:13:40 2022

@author: xijiaqi
"""

import ot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

sample_sizes = [1000,3000,5000,10000]

m = 1

n_seed = 10
ns = 500
for n in sample_sizes:
    swd = np.empty((ns,))
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    for i in range(ns):
        smp = np.empty((n_seed,))
        datap = generate_uniform_sphere(d,n,Rp)
        dataq = generate_uniform_sphere(d,n,Rq)       
        for seed in range(n_seed):
            smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
        swd[i] = np.mean(smp)
    swd = np.sqrt(n)*(swd - rwpp)
    density = gaussian_kde(swd,'silverman')
    plt.figure(m)
    plt.plot(xs,density(xs),color='cadetblue')
    plt.fill_between(xs, density(xs),color='paleturquoise',alpha=0.5)
    plt.plot(xs,realden(xs),color='palevioletred')
    plt.fill_between(xs,realden(xs),color='pink',alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title('sample size n = '+str(n))
    m += 1