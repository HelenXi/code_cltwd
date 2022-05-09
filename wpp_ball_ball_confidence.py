#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:56:50 2022

@author: xijiaqi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import pylab
from PRW import ProjectionRobustWasserstein
from Optimization.riemann_adap import RiemmanAdaptive

def ball_ball(n,d,R,dim=3):

    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    # First measure : uniform on the unit disk
    angles = np.random.randn(n,dim)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    radii = (np.random.uniform(0,1,size=n))**(1./dim)*R
    X = np.diag(radii).dot(angles)
    X = np.append(X, np.random.uniform(size=(n,d-dim)), axis=1)

    # Second measure : uniform on the annulus 2≤r≤3
    angles = np.random.randn(n,dim)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    
    #radii = ((3**dim - 1**dim)*np.random.uniform(0, 1, size=n) + 1**dim)**(1./dim)
    radii = (np.random.uniform(0,1,size=n))**(1./dim)
    Y = np.diag(radii).dot(angles)
    Y = np.append(Y, np.random.uniform(size=(n,d-dim)), axis=1)
    
    return a,b,X,Y

n = 800
d = 20
k = 3
l = 32
B = 1000
al, bl = np.ones((l,)) / l, np.ones((l,)) / l
alpha = 0.05

xs = np.linspace(-5,5,200)

cost = 3/5
var = 12/7-36/25+3/7-9/25 
realden = np.exp(-xs**2/(2*var))/np.sqrt(2*np.pi*var)

ns = 200

values = np.zeros((ns,))
lc_list = np.zeros((ns,))
rc_list = np.zeros((ns,))

for i in range(ns):
    a,b,X,Y = ball_ball(n, d, 1.7)
    algo = RiemmanAdaptive(reg=0.2, step_size_0=None, max_iter=30, threshold=1e-4, \
                           max_iter_sinkhorn=30,threshold_sinkhorn=1e-04, use_gpu=False)
    PRW = ProjectionRobustWasserstein(X, Y, a, b, algo, k)
    Omega, pi, maxmin_values = PRW.run(1, lr=0.01, beta=0.8)
    values[i] = PRW.get_value()
    boot = np.empty((B,))
    for j in range(B):
        indices = np.random.choice(n,l,replace = True)
        Xrep = X[indices]
        Yrep = Y[indices]
        algo_rep = RiemmanAdaptive(reg=0.2, step_size_0=None, max_iter=30, threshold=1e-4, \
                           max_iter_sinkhorn=30,threshold_sinkhorn=1e-04, use_gpu=False)
        PRW_rep = ProjectionRobustWasserstein(Xrep, Yrep, al, bl, algo_rep, k)
        Omega_rep, pi_rep, maxmin_values_rep = PRW_rep.run(0, lr=0.01, beta=None)
        boot[j] = PRW_rep.get_value()
    p_alpha = np.percentile(boot,2.5)
    q_alpha = np.percentile(boot,97.5)
    lc = values[i] - q_alpha/np.sqrt(n)
    rc = values[i] + p_alpha/np.sqrt(n)
    lc_list[i] = lc
    rc_list[i] = rc


values_mean = np.mean(values)
values_10 = np.percentile(values, 10)
values_25 = np.percentile(values, 25)
values_75 = np.percentile(values, 75)
values_90 = np.percentile(values, 90)

values = np.sqrt(n)*(values - values_mean)
values_std = np.std(values)



    
    
    