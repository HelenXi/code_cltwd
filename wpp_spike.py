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

def generate_uniform_sphere(d,n):
    data = np.zeros((n,d))
    for j in range(n):
        temp = np.random.normal(0,1,size=(1,d))
        data[j] = temp/np.linalg.norm(temp)
    return data

def generate_uniform_cube(d,n):
    data = np.zeros((n,d))
    for j in range(n):
        data[j] = np.random.rand(1,d)*2-1
    return data

def generate_uniform_ball(d,n,R):
    data = np.zeros((n,d))
    for j in range(n):
        temp = np.random.normal(0,1,size=(1,d))
        r = np.power(R*np.random.rand(),1/d)
        data[j] = temp*r
    return data

def generate_uniform_ellipsoid(d,n,sigma,v):
    data = np.zeros((n,d))
    scale = np.ones((d,))
    scale[v] = sigma
    for j in range(n):
        temp = np.random.normal(0,scale)
        data[j] = temp/np.sqrt(np.linalg.norm(temp)**2 +(1/sigma**2-1)*temp[v]**2)
    return data


d = 3
v = 1 #np.random.randint(1,d)
sigma = 8.512553358247274 #np.random.rand()*10
rwpp = (sigma-1)**2/3
rvar = 4*(1-sigma)**2*(sigma**2+1)/45


sample_sizes = [50,100,500]

m = 1
xs = np.linspace(-100,100,1000)
limSdens = np.exp(-xs**2/(2*rvar))/np.sqrt(2*rvar*np.pi)

n_seed = 50
for n in sample_sizes:
    swd = np.empty((2000,))
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    for i in range(2000):
        smp = np.empty((n_seed,))
        datap = generate_uniform_ellipsoid(d,n,sigma,v)
        dataq = generate_uniform_sphere(d,n)       
        for seed in range(n_seed):
            smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
        swd[i] = np.mean(smp)
    swd = np.sqrt(n)*(swd - rwpp)
    density = gaussian_kde(swd,'silverman')
    plt.figure(m)
    plt.plot(xs,density(xs),color='cadetblue')
    plt.fill_between(xs, density(xs),color='paleturquoise',alpha=0.5)
    plt.plot(xs,limSdens,color='palevioletred')
    plt.fill_between(xs,limSdens,color='pink',alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title('sample size n = '+str(n))
    m += 1