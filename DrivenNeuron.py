# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:15:11 2016

@author: maria
"""
from dana import * 
import numpy as np
import matplotlib.pyplot as plt

'''
drive=50
tau=10
neuron=Group((1,),'''#dx/dt=(-x+drive)/tau : float
                                #''')

def exponentiate(t):
    t=-(float(t)/20)
    exp=np.exp(t)
    return exp

t=[]
z=0
for j in range(0,10002):
    z=z+0.01
    t.append(z)
    
neuron2=neuron=Group((1,),'''dx/dt=(-x+40*2.7183**(-t/20))/20 : float
                                t:float''')

rate=[]
rate2=[] 
global r
r=0                          
@clock.every(0.01)
def record(*args):
    global r
    #rate.append(neuron['x'][0])
    neuron2['t']=t[r]
    r=r+1
    rate2.append(neuron2['x'][0])
    
run(time=100*second,dt=0.01*second)

plt.plot(rate2)
