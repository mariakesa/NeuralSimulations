# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:54:45 2017

@author: maria
"""

from dana import * 
import numpy as np
import matplotlib.pyplot as plt


n_of_l4_neurons=100
n_of_l23_neurons=100

def draw_connectivity(n_of_l4_neurons,n_of_l23_neurons):
    '''
    This function implements the connectivity between l4 and l2/3 cells,
    which determines which input go to which cells. The activations
    that don't contribute toward a l2/3 cell response are 0.
    '''
    connectivity=np.random.binomial(1,0.2,(n_of_l4_neurons,n_of_l23_neurons))
    return connectivity
    
def draw_w0(n_of_l4_neurons,n_of_l23_neurons):
    w0=np.abs(np.random.normal(0,1,(n_of_l4_neurons,n_of_l23_neurons)))
    return w0
    
    
C=draw_connectivity(n_of_l4_neurons,n_of_l23_neurons)
w0=draw_w0(n_of_l4_neurons,n_of_l23_neurons)

#print C,w0

def generate_stimulus_to_l23(n_of_l4_neurons):
    '''
    This function implements the activations in l4 cells that are
    used as input to layer 2/3.
    '''
    stimulus=[]
    std=1
    stimulus_A=np.random.normal(0,std,n_of_l4_neurons)
    stimulus_B=np.random.normal(0,std,n_of_l4_neurons)
    stimulus_C=np.random.normal(0,std,n_of_l4_neurons)
    stimulus_D=np.random.normal(0,std,n_of_l4_neurons)
    for j in range(0,5):
        for time_step in range(0,250):
            stimulus.append(stimulus_A)
        for time_step in range(0,250):
            stimulus.append(stimulus_B)
        for time_step in range(0,250):
            stimulus.append(stimulus_C)
        for time_step in range(0,250):
            stimulus.append(stimulus_D)
    stimulus_X=np.random.normal(0,std,n_of_l4_neurons)
    for time_step in range(0,250):
        stimulus.append(stimulus_A)
    for time_step in range(0,250):
        stimulus.append(stimulus_B)
    for time_step in range(0,250):
        stimulus.append(stimulus_C)
    for time_step in range(0,250):
        stimulus.append(stimulus_X) 
    return stimulus
    
def generate_mask(n_of_l4_neurons,n_of_l23_neurons):
    mask_depr=np.random.binomial(1, 0.9, (n_of_l4_neurons,n_of_l23_neurons))
    return mask_depr
    
mask_depr=generate_mask(n_of_l4_neurons,n_of_l23_neurons)
mask_facil=np.zeros((n_of_l4_neurons,n_of_l23_neurons))
for i in range(0,n_of_l4_neurons):
    for j in range(0,n_of_l23_neurons):
        if mask_depr[i,j]==0:
            mask_facil[i,j]=1
        else:
            mask_facil[i,j]=0

stimulus=generate_stimulus_to_l23(n_of_l4_neurons)

alpha=0.04
beta=-0.15
tau_depr=0.094*second
tau_facil=0.038*second
tau_cell=0.5*second
f_max=2
#print mask_depr, mask_facil
l4=np.ones((n_of_l4_neurons,))

def F(weights):
    for j in range(0, len(weights)):
        if weights[j]<0:
            weights[j]=0
    return weights
    

response=Group((n_of_l23_neurons,),'''dr/dt=(-r+F(weighted_input))/tau_cell :float
                        weighted_input:float
                        ''')
facil_depr=Group((n_of_l4_neurons,n_of_l23_neurons),
                 '''dW/dt = (1-W-alpha*W*l4*mask_depr+beta*(f_max-W)*l4*mask_facil)/(mask_depr*tau_depr+mask_facil*tau_facil):float
                    l4:float''')
baseline=w0*C
weights=np.ones((n_of_l4_neurons,n_of_l23_neurons))

con =DenseConnection(l4, response('weighted_input'), weights)
 
response_over_time=[]
for j in range(0,n_of_l23_neurons):
    response_over_time.append([])

t=0
@clock.every(0.01)
def record_weights(*args):
    global t
    l4[:]=stimulus[t]
    facil_depr['l4']=np.reshape(np.repeat(stimulus[t],n_of_l23_neurons),(n_of_l4_neurons,n_of_l23_neurons))
    weights[:,:]=baseline*facil_depr['W']
    for neuron in range(0,n_of_l23_neurons):
        response_over_time[neuron].append(response['r'][neuron])
    t=t+1

run(time=59.99*second,dt=0.01*second)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('Cell rates in layer 2/3, Feed-forward model')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Rate')
for j in range(0,5):
    fig=plt.plot(response_over_time[j])
    


    

