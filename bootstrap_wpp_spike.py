#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 10:34:29 2022

@author: xijiaqi
"""

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
from fractions import Fraction

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


n = 1000
B = 500
replacement = [1,7/8,3/4,1/2]

xs = np.linspace(-100,100,1000)
limSdens = np.exp(-xs**2/(2*rvar))/np.sqrt(2*rvar*np.pi)

n_seed = 50
swd = np.empty((200,))
boot = np.empty((500,))
a, b = np.ones((n,)) / n, np.ones((n,)) / n

m = 1
for li in replacement:
    l = int(np.power(n,li))
    al, bl = np.ones((l,)) / l, np.ones((l,)) / l
    for i in range(200):
        smp = np.empty((n_seed,))
        btp = np.empty((n_seed,))
        datap = generate_uniform_ellipsoid(d,n,sigma,v)
        dataq = generate_uniform_sphere(d,n)       
        for seed in range(n_seed):
            smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
        swd[i] = np.mean(smp)
    datap = generate_uniform_ellipsoid(d,n,sigma,v)
    dataq = generate_uniform_sphere(d,n)
    for seed in range(n_seed):
            smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
    swd_boot = np.mean(smp)
    for j in range(B):
        indices = np.random.choice(n,l,replace = True)
        rep = datap[indices]
        req = dataq[indices]
        for seed in range(n_seed):
            btp[seed] = ot.sliced.max_sliced_wasserstein_distance(rep, req, al, bl, 1000, seed=seed)**2
        boot[j] = np.sqrt(l)*(np.mean(btp) - swd_boot)
    swd = np.sqrt(n)*(swd - rwpp)
    densswd = gaussian_kde(swd,'silverman')
    densboot = gaussian_kde(boot,'silverman')
    plt.figure(m)
    plt.plot(xs,densswd(xs),color='cadetblue')
    plt.fill_between(xs,densswd(xs),color='paleturquoise',alpha=0.3)
    plt.plot(xs,densboot(xs),color = 'darkolivegreen')
    plt.fill_between(xs,densboot(xs),color='palegreen',alpha = 0.4)
    plt.plot(xs,limSdens,color='palevioletred')
    plt.fill_between(xs,limSdens,color='pink',alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title('replacement l = n^'+str(Fraction(li)))
    m += 1




