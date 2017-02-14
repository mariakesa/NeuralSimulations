# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:49:47 2017

@author: maria
"""
'''
This program makes sure that the weights obey Dale's law, e.g. inhibitory neurons don't emit
excitatory connections.
'''
def generate_weights(n_of_neurons,n_of_exc,n_of_inh):
    weights=np.zeros((n_of_neurons,n_of_neurons))
    weights[0:n_of_exc,0:n_of_exc]=np.random.binomial(0.1,1,(n_of_exc,n_of_exc))
    weights[0:n_of_exc,n_of_exc:n_of_exc+n_of_inh]=-np.random.binomial(0.1,1,(n_of_exc,n_of_inh))
    weights[n_of_exc:n_of_exc+n_of_inh,0:n_of_exc]=np.random.binomial(0.1,1,(n_of_inh,n_of_exc))
    '''
    weights[0:n_of_exc,0:n_of_exc]=exc_to_exc
    weights[0:n_of_exc,n_of_exc:n_of_exc+n_of_inh]=inh_to_exc
    weights[n_of_exc:n_of_exc+n_of_inh,0:n_of_exc]=exc_to_inh
    #weights[n_of_exc:n_of_exc+n_of_inh,n_of_exc:n_of_exc+n_of_inh]=inh_to_inh
    '''
    return weights

from dana import * 
import numpy as np
import matplotlib.pyplot as plt


n_of_inh_neurons=100
n_of_pyr_neurons=100
n_of_neurons=n_of_inh_neurons+n_of_pyr_neurons
tau_cell=0.2

#Stimulus protocol

def generate_stimulus_to_excitatory(nr_of_exc_neurons,nr_of_inh_neurons):
    stimulus=[]
    std=10
    stimulus_A=np.random.normal(0,std,nr_of_exc_neurons)
    stimulus_A=np.append(stimulus_A, np.zeros(nr_of_inh_neurons))
    stimulus_B=np.random.normal(0,std,nr_of_exc_neurons)
    stimulus_B=np.append(stimulus_B, np.zeros(nr_of_inh_neurons))
    stimulus_C=np.random.normal(0,std,nr_of_exc_neurons)
    stimulus_C=np.append(stimulus_C, np.zeros(nr_of_inh_neurons))
    stimulus_D=np.random.normal(0,std,nr_of_exc_neurons)
    stimulus_D=np.append(stimulus_D, np.zeros(nr_of_inh_neurons))
    for j in range(0,5):
        for time_step in range(0,250):
            stimulus.append(stimulus_A)
        for time_step in range(0,250):
            stimulus.append(stimulus_B)
        for time_step in range(0,250):
            stimulus.append(stimulus_C)
        for time_step in range(0,250):
            stimulus.append(stimulus_D)
    stimulus_X=np.random.normal(0,std,nr_of_exc_neurons)
    stimulus_X=np.append(stimulus_X, np.zeros(nr_of_inh_neurons))
    for time_step in range(0,250):
        stimulus.append(stimulus_A)
    for time_step in range(0,250):
        stimulus.append(stimulus_B)
    for time_step in range(0,250):
        stimulus.append(stimulus_C)
    for time_step in range(0,250):
        stimulus.append(stimulus_X) 
    return stimulus

stimulus=generate_stimulus_to_excitatory(n_of_pyr_neurons,n_of_inh_neurons)

def F(weights):
    for j in range(0, len(weights)):
        if weights[j]<0:
            weights[j]=0
    return weights
    
def sigmoid(x):
    r_max=1
    g=1
    return float(r_max)/ (1 + np.exp(-x))
    
weights=np.load('soc.txt.npy')
#weights=generate_weights(n_of_neurons, n_of_pyr_neurons,n_of_inh_neurons)

neurons=Group((n_of_neurons,),'''dR/dt=(-R+F(I+weighted_combination+10))/tau_c : float
                                weighted_combination:float
                                I: float
                                dT/dt = (R**2-T)*1/tau_t : float
                                ''')
                                
tau_c = 1.0
tau_t = tau_c * 0.1
eta   = tau_t * 0.1

A=DenseConnection(neurons.R,neurons('weighted_combination'),weights,
                             'dW/dt=pre*post.R*(post.R-post.T)*eta')
#C = Connection(source, target, kernel)
neurons.R=np.ones((n_of_neurons,))

rate=[]
for j in range(0,n_of_neurons):
    rate.append([])
global t
t=0
#w=[]
combination=[]
for j in range(0,n_of_neurons):
    combination.append([])
I=np.zeros((n_of_neurons,))
@clock.every(0.01)
def record_weights(*args):
    global t, rate
    neurons['I'][:]=stimulus[t]
    for neuron in range(0,n_of_neurons):
        rate[neuron].append(neurons['R'][neuron])
        combination[neuron].append(neurons['weighted_combination'][neuron])
    #Set excitatory cells to be positive
    for index1 in range(0,n_of_neurons):
        for index2 in range(0,n_of_pyr_neurons):
            if A.weights[index1,index2]<0:
                A.weights[index1,index2]=0
    #0:n_of_exc,n_of_exc:n_of_exc+n_of_inh
    #Force inh cells to exc cells to be negative
    for index1 in range(0,n_of_pyr_neurons):
        for index2 in range(n_of_pyr_neurons,n_of_pyr_neurons+n_of_inh_neurons):
            if A.weights[index1,index2]>0:
                A.weights[index1,index2]=0
    #Set inh-inh connections
    #weights[n_of_exc:n_of_exc+n_of_inh,n_of_exc:n_of_exc+n_of_inh]=inh_to_in
    for index1 in range(n_of_pyr_neurons,n_of_pyr_neurons+n_of_inh_neurons):
        for index2 in range(n_of_pyr_neurons,n_of_pyr_neurons+n_of_inh_neurons):
            if A.weights[index1,index2]>0:
                A.weights[index1,index2]=0
    t=t+1
    #w.append(connectivity._weights[0,1])
    
#run(time=109.99*second,dt=0.01*second)
run(time=59.99*second,dt=0.01*second)

mean=[]
for j in range(0,len(rate[0])):
    zum=0
    for neuron in range(0,100):
        zum+=rate[neuron][j]
    mean.append(zum)
        
plt.plot(mean)

'''
for j in range(0,100):
    plt.plot(rate[j][8000:])
'''
    