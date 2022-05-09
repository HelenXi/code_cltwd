#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:47:01 2022

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

def generate_uniform_ellipsoid(d,n,sigma):
    data = np.zeros((n,d))
    for j in range(n):
        temp = np.random.normal(0,sigma)
        data[j] = temp/np.sqrt(sum(temp**2/sigma**2))
    return data

    

Rp = 1
Rq = 2
d = 3


sigma = np.array([2,0.5,1])

n = 1000

ns = 1000
N = 2000
m = 1
xs = np.linspace(-2,2,400)
swd = np.empty((ns,))
t = 1/4

'''fraction = 0
for j in range(N):
    u = generate_uniform_sphere(d,1,1)
    u = u[0]
    if (1 - np.sqrt(u[0]**2*4 + u[1]**2/4+u[2]**2))**2 > 3*t:
        fraction += 1
fraction /= N

for i in range(ns):
    datap = generate_uniform_sphere(d,n,1)
    dataq = generate_uniform_ellipsoid(d,n,sigma)
    smp = np.zeros((N,))
    count = 0
    for j in range(N):
        u = generate_uniform_sphere(d,1,1)
        u = u[0]
        pu = np.dot(datap,u)
        qu = np.dot(dataq,u)
        if ot.emd2_1d(pu,qu,a,b) > t:
           count += 1
    swd[i] = np.sqrt(n)*(count/N - fraction)
swd_mean = np.mean(swd)
swd_std = np.std(swd)
density = gaussian_kde(swd,'silverman')
plt.figure(m)
plt.plot(xs,density(xs),color='navy')
plt.fill_between(xs, density(xs),color='mediumpurple',alpha=0.5)
plt.xlabel("x")
plt.ylabel("Density")
plt.title('t = 1/4')'''

n = 600
a, b = np.ones((n,)) / n, np.ones((n,)) / n
ns = 500
amp = np.empty((ns,))
rvar = 4*(1-2)**2*(2**2+1)/45 + 4*(1-4)**2*(4**2+1)/45 - 2*(1/5-1/3)*9
#4*(1-3)**2*(3**2+1)/45 + 4*(1-0.5)**2*(0.5**2+1)/45 - 2*(1/9+1/6-1/15-3/20) (2,0.5,3)
xs = np.linspace(-15,15,2000)
limSdens = np.exp(-xs**2/(2*rvar))/np.sqrt(2*rvar*np.pi)
n_seed = 20
for i in range(ns):
    datap = generate_uniform_ellipsoid(d,n,np.array([2,2,4]))
    dataq = generate_uniform_sphere(d,n,1)       
    smp = np.empty((n_seed,))
    '''for seed in range(n_seed):
        smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
    amp[i] = np.mean(smp)'''
    amp[i] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=np.random.randint(50))**2
    v0 = 1/3
    angles = generate_uniform_sphere(d,N,1)
    for j in range(N):
        u = angles[j]
        pu = np.dot(datap,u)
        qu = np.dot(dataq,u)
        v0 = min(v0,ot.emd2_1d(pu,qu,a,b))
    amp[i] -= v0
amp = np.sqrt(n)*(amp - 8/3) 
density = gaussian_kde(amp,'silverman')
plt.plot(xs,density(xs),color='cadetblue')
plt.fill_between(xs, density(xs),color='paleturquoise',alpha=0.5)
plt.plot(xs,limSdens,color='palevioletred')
plt.fill_between(xs,limSdens,color='pink',alpha=0.5)
plt.xlabel("x")
plt.ylabel("Density")
plt.title('amplitude')


    
        
        
