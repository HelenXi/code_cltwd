#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:11:29 2022

@author: xijiaqi
"""

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
vaS = 0.8269 #5/9

n = 1000
B = 500
replacement = [1,7/8,3/4,1/2]

m = 1
xs = np.linspace(-5,5,200)
limSdens = np.exp(-xs**2/(2*vaS))/np.sqrt(2*vaS*np.pi)

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
        datap = generate_uniform_sphere(d,n,Rq)+1
        dataq = generate_uniform_sphere(d,n,Rq)     
        for seed in range(n_seed):
            smp[seed] = ot.sliced.sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
        swd[i] = np.mean(smp)
    datap = generate_uniform_sphere(d,n,Rq)+1
    dataq = generate_uniform_sphere(d,n,Rq)
    for seed in range(n_seed):
            smp[seed] = ot.sliced.sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
    swd_boot = np.mean(smp)
    for j in range(B):
        indices = np.random.choice(n,l,replace = True)
        rep = datap[indices]
        req = dataq[indices]
        for seed in range(n_seed):
            btp[seed] = ot.sliced.sliced_wasserstein_distance(rep, req, al, bl, 1000, seed=seed)**2
        boot[j] = np.sqrt(l)*(np.mean(btp) - swd_boot)
    swd = np.sqrt(n)*(swd - rswd)
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




