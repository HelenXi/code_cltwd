#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:19:55 2022

@author: xijiaqi
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import statsmodels.api as sm
import scipy.stats as stats
import ot



def disk_annulus(n,d):

    a = (1./n) * np.ones(n)
    b = (1./n) * np.ones(n)

    # First measure : uniform in the unit disk
    angles = np.random.randn(n,d)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    radii = (np.random.uniform(0,1,size=n))
    X = np.diag(radii).dot(angles)

    # Second measure : uniform in the non-unit disk
    angles = np.random.randn(n,d)
    angles = (angles.T/np.linalg.norm(angles, axis=1)).T
    radii = (np.random.uniform(0,1,size=n))
    Y = np.diag(radii).dot(angles)
    
    return a,b,X,Y

n = 800
d = 2
k = 1
ns = 300
n_seed = 50

xs = np.linspace(-5,5,200)
values = np.zeros((50,50))
angle_list = np.zeros((50,2))
angle_list[:,0] = np.cos(np.linspace(0,2*np.pi,50))
angle_list[:,1] = np.sin(np.linspace(0,2*np.pi,50))

for i in range(50):
    u = angle_list[i]
    for j in range(50):
        v = angle_list[j]
        temp = 0
        for n in range(ns):
            a,b,X,Y = disk_annulus(n,d)
            Xu, Xv = np.inner(X,u), np.inner(X,v)
            Yu, Yv = np.inner(Y,u), np.inner(Y,v)
            temp += ot.emd2_1d(Xu,Yu,a,b)*ot.emd2(Xv,Yv,a,b)
        values[i,j] = n*temp/ns

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot3D(values, 'green')
ax.set_title('covariance function K(u,v)')
plt.show()

    
    
