# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 05:10:35 2016

@author: maria
"""

from dana import * 
import numpy as np
import matplotlib.pyplot as plt

#Set parameters
n_of_sensory_neurons=4
n_of_inh_neurons=3
n_of_pyr_neurons=9
n_of_neurons=n_of_inh_neurons+n_of_pyr_neurons
tau_cell=1

def generate_sensory_signal(n_of_neurons):
    #sensory_rate=[100, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0]
    all_sensory=[]
    for j in range(0,10002):
        all_sensory.append([100, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0, 0])
    return all_sensory

def generate_weights(n_of_exc,n_of_inh):
    '''
    The function assumes that the first 'n_of_exc' entries in the
    rate vector correspond to excitatory neurons and the last 'n_of_inh'
    entries correspond to inhibitory neurons.
    '''
    weights=np.zeros((n_of_exc+n_of_inh,n_of_exc+n_of_inh))
    exc_to_exc=np.ones((n_of_exc,n_of_exc))
    inh_to_exc=-np.ones((n_of_exc,n_of_inh))
    exc_to_inh=np.ones((n_of_inh,n_of_exc))
    inh_to_inh=-np.ones((n_of_inh,n_of_inh))
    weights[0:n_of_exc,0:n_of_exc]=exc_to_exc
    weights[0:n_of_exc,n_of_exc:n_of_exc+n_of_inh]=inh_to_exc
    weights[n_of_exc:n_of_exc+n_of_inh,0:n_of_exc]=exc_to_inh
    weights[n_of_exc:n_of_exc+n_of_inh,n_of_exc:n_of_exc+n_of_inh]=inh_to_inh
    #Set self-connections to zero
    for index in range(0,n_of_exc+n_of_inh):
        weights[index,index]=0
    return weights
    


def F(weights):
    for j in range(0, len(weights)):
        if weights[j]<0:
            weights[j]=0
    return weights
    
weights=generate_weights(n_of_pyr_neurons,n_of_inh_neurons)
all_sensory=generate_sensory_signal(n_of_neurons)

neurons=Group((n_of_neurons,),'''dR/dt=(-R+F(I+weighted_combination))/tau_cell : float
                                weighted_combination:float
                                ''')
connectivity=DenseConnection(neurons('R'),neurons('weighted_combination'),weights)
#C = Connection(source, target, kernel)
neurons.R=np.ones((n_of_neurons,))

rate=[]
for j in range(0,n_of_neurons):
    rate.append([])
global t
t=0
#w=[]
I=[0,0,0,0,0,0,0,0,0,0,0,0]
@clock.every(0.01)
def record_weights(*args):
    global t, rate
    I[:]=all_sensory[t]
    for neuron in range(0,n_of_neurons):
        rate[neuron].append(neurons['R'][neuron])
    #w.append(connectivity._weights[0,1])
    
run(time=100*second,dt=0.01*second)

for j in range(0,n_of_neurons):
    plt.plot(rate[j][0:3000])
#plt.plot(w)


    