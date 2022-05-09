#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:32:19 2022

@author: xijiaqi
"""

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


def sliced(datap,dataq,a,b,npro):
    n = len(datap)
    sl = 0
    for i in range(npro):
        v = generate_uniform_sphere(d,1,1)
        v = v[0]
        tmp = np.array([np.dot(datap[j],v) for j in range(n)])
        tmq = np.array([np.dot(dataq[j],v) for j in range(n)])
        sl += ot.wasserstein_1d(tmp,tmq,a,b,p=2)
    return sl/npro
    

Rp = 5
Rq = 1
d = 3
rswd = 1
vaS = 0.8269


sample_sizes = [50,100,500]

m = 1
xs = np.linspace(-5,5,200)
limSdens = np.exp(-xs**2/(2*vaS))/np.sqrt(2*vaS*np.pi)

n_seed = 50
for n in sample_sizes:
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    swd = np.empty((100,))
    for i in range(100):
        datap = generate_uniform_sphere(d,n,Rq)+1
        dataq = generate_uniform_sphere(d,n,Rq)
        smp = np.empty((n_seed,))
        for seed in range(n_seed):
            smp[seed] = ot.sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
        swd[i] = np.mean(smp)
    swd_mean = np.mean(swd)
    swd = np.sqrt(n)*(swd - rswd)     
    swd_std = np.std(swd)
    density = gaussian_kde(swd,'silverman')
    plt.figure(m)
    plt.plot(xs,density(xs),color='cadetblue')
    plt.fill_between(xs, density(xs),color='paleturquoise',alpha=0.5)
    limSdense = np.exp(-xs**2/(2*swd_std**2))/np.sqrt(2*np.pi*swd_std**2)
    plt.plot(xs,limSdens,color='palevioletred')
    plt.fill_between(xs,limSdens,color='pink',alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title('sample size n = '+str(n))
    m += 1
    
        
        
