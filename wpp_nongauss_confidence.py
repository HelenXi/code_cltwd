#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 00:46:16 2022

@author: xijiaqi
"""

import numpy as np
import ot
import matplotlib.pyplot as plt
from fractions import Fraction

def generate_uniform_sphere(d,n,R):
    data = np.zeros((n,d))
    for j in range(n):
        temp = np.random.normal(0,1,size=(1,d))
        data[j] = R*temp/np.linalg.norm(temp)
    return data





n = 1000
d = 3
k = 1
B = 500
alpha = 0.05
ns = 200
Rp = 1
Rq = 2
rwpp = 1/3
n_seed = 50
replacement = [1,7/8,3/4,1/2]
values = np.zeros((ns,))
lc_list = np.zeros((4,ns))
rc_list = np.zeros((4,ns))
rej_rate = np.zeros((4,))
a, b = np.ones((n,)) / n, np.ones((n,)) / n
smp = np.empty((n_seed,))

dataps = np.empty((ns,n,d))
dataqs = np.empty((ns,n,d))
for i in range(ns):
    datap = generate_uniform_sphere(d, n, Rp)
    dataq = generate_uniform_sphere(d, n, Rq)
    dataps[i] = datap
    dataqs[i] = dataq
    for seed in n_seed:
        smp[seed] = ot.sliced.max_sliced_wasserstein_distance(datap, dataq, a, b, 1000, seed=seed)**2
    values[i] = np.mean(smp)
    
for li in replacement:
    l = int(np.power(n,li))
    al, bl = np.ones((l,)) / l, np.ones((l,)) / l
    m = replacement.index(li)
    rej_count = 0
    
    for i in range(ns):
        boot = np.empty((B,))
        val = 1
        datap = dataps[i]
        dataq = dataqs[i]
        for r in range(B):
            indices = np.random.choice(n,l,replace = True)
        rep = datap[indices]
        req = dataq[indices]
        for seed in n_seed:
            smp[seed] = ot.sliced.max_sliced_wasserstein_distance(rep, req, al, bl, 1000, seed=seed)**2
        boot[r] = np.mean(smp)
        p_alpha = np.percentile(boot,2.5)
        q_alpha = np.percentile(boot,97.5)
        lc = values[i] - q_alpha/np.sqrt(n)
        rc = values[i] - p_alpha/np.sqrt(n)
        if not lc <= rwpp <= rc:
            rej_count += 1
        lc_list[m,i] = lc
        rc_list[m,i] = rc
    rej_rate[m] = rej_count/ns

l = len(replacement)    
plt.figure(1)
plt.plot(range(l),rej_rate, color = 'orchid',marker="o")
plt.xticks(range(l),[Fraction(li) for li in replacement])
plt.xlabel("replacement")
plt.title("Rejection Rates")
plt.show()

lc_avg = [np.mean(lc_list[i,:]) for i in range(4)]
rc_avg = [np.mean(rc_list[i,:]) for i in range(4)]
cmean = [(lc_avg[i] + rc_avg[i])/2 for i in range(4)]
uperr = [rc_avg[i] - cmean[i] for i in range(4)]
loerr = [cmean[i] - lc_avg[i] for i in range(4)]
plt.figure(2)
plt.errorbar(range(l),cmean,yerr = np.array([uperr,loerr]),capsize=15, elinewidth=1.3, markeredgewidth=1.5,ecolor='black',color='rosybrown',linewidth=2.5)
plt.xticks(range(l),[Fraction(li) for li in replacement])
plt.xlabel("replacement")
plt.title("Confidence Intervals")
plt.show()