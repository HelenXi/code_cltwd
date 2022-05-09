#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:22:08 2022

@author: xijiaqi
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import scipy.stats as stats

from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive



def ball_ball(n,d,dim=3):

    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    # First measure : uniform on the unit disk
    angles = np.random.randn(n,dim)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    radii = (np.random.uniform(0,1,size=n))**(1./dim)
    X = np.diag(radii).dot(angles)
    X = np.append(X, np.random.uniform(size=(n,d-dim)), axis=1)

    # Second measure : uniform on the annulus 2≤r≤3
    angles = np.random.randn(n,dim)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    
    #radii = ((3**dim - 1**dim)*np.random.uniform(0, 1, size=n) + 1**dim)**(1./dim)
    radii = (np.random.uniform(0,1,size=n))**(1./dim)*2
    Y = np.diag(radii).dot(angles)
    Y = np.append(Y, np.random.uniform(size=(n,d-dim)), axis=1)
    
    return a,b,X,Y

n = 500
d = 20
k = 3
l = 250
B = 1000
al, bl = np.ones((l,)) / l, np.ones((l,)) / l
alpha = 0.05

xs = np.linspace(-5,5,200)

cost = 3/5
var = 12/7-36/25+3/7-9/25 
realden = np.exp(-xs**2/(2*var))/np.sqrt(2*np.pi*var)

n_list = [50,100,500,1000,2000] 
ns = 200

values = np.zeros((len(n_list),ns))

for j in range(len(n_list)):
    n = n_list[j]
    for i in range(ns):
        a,b,X,Y = ball_ball(n, d)
        algo = RiemmanAdaptive(reg=0.2, step_size_0=None, max_iter=30, threshold=1e-4, \
                           max_iter_sinkhorn=30,threshold_sinkhorn=1e-04, use_gpu=False)
        PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
        Omega, pi, maxmin_values = PRW.run(0, lr=0.01, beta=None)
        values[j,i] = PRW.get_value() - 0.2    
    values[j,:] = np.sqrt(n)*(values[j,:] - np.mean(values[j,:]))
    
m = 1
for j in range(len(n_list)):
    plt.figure(m)
    sm.qqplot(values[j,:]/np.std(values[j,:]),line='45')
    plt.title('sample size n = ' + str(n_list[j]))
    m += 1

ks_dis = np.zeros((5,2))
for j in range(len(n_list)):
    ks_dis[j] = stats.ks_2samp(values[j,:], np.random.normal(0,np.sqrt(var),size=(ns,)))
    
plt.figure(m)
plt.plot(n_list,ks_dis[:,0],color='coral',marker='o',linestyle='-')
plt.xlabel('sample size n')
plt.title('Kolmogorov-Smirnov distance')
plt.show()
    
    
    


