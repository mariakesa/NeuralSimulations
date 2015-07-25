# -*- coding: utf-8 -*-
"""
Created on Tue May 19 01:37:39 2015

@author: maria
"""

from brian import *
import pyentropy
import matplotlib.pyplot as plt

eqs_i='''
dv/dt=(-gl_i*(v-El_i)-g_ext_i*s_ext*(v-E_ampa))/Cm_i: volt
ds_ext/dt=-s_ext/t_ampa : 1
'''

defaultclock.dt = 0.01*ms
defaultclock.reinit()
simulation_time=0.5*second

nr_of_neurons=10
t_ampa=2*ms
gl_i=25*nS
El_i = -70*mV
E_ampa=0*mV
g_ext_i=3.1*nS
Cm_i=0.5*nF
Vth_i=-50*mV
Vr_i=-60*mV
tr_i = 2*ms
fext=1.8*kHz
neurons = NeuronGroup(nr_of_neurons, eqs_i,threshold=Vth_i, reset=Vr_i,refractory=tr_i)

inputs = PoissonGroup(nr_of_neurons,fext)

input_connections = IdentityConnection(inputs,neurons,'s_ext',weight=1.0)

M = SpikeMonitor(neurons)
run(simulation_time)
spikes=M.spikes
#raster_plot(M)

def generate_empty_bins(nr_of_bins,nr_of_neurons):
    spike_list=[]
    for neuron in range(0,nr_of_neurons):
        bin_list=[]
        for time_bin in range(0,nr_of_bins):
            bin_list.append(0)
        spike_list.append(bin_list)
    return spike_list

def discretize_spikes(nr_of_bins,time_interval,spikes,nr_of_neurons):
    time_bin=float(time_interval)/nr_of_bins
    time_bin_state=time_bin*second
    state=0
    spike_list=generate_empty_bins(nr_of_bins,nr_of_neurons)
    for spike in spikes:
        spike_index=spike[0]
        #This line makes the code work, but I have to look into Brian more in
        #depth, something seems to be wrong with the timing
        spike_time=spike[1]
        if spike_time>time_bin_state:
            time_bin_state=time_bin_state+time_bin*second
            state+=1
            spike_list[spike_index][state]=1
        else:
            spike_list[spike_index][state]=1
    return spike_list
    
nr_of_bins=100
spike_list=discretize_spikes(nr_of_bins,simulation_time,spikes,nr_of_neurons)

    
def calculate_mutual_information(X,Y,n):
    s=pyentropy.systems.DiscreteSystem(X,(1,n), Y,(1,n))
    s.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
    return s.I()

lst=[]
for i in range(0,nr_of_neurons):
    mutual_information_list=[]
    for j in range(0,nr_of_neurons):
        mut_inf = calculate_mutual_information(spike_list[i],spike_list[j],nr_of_bins)           
        mutual_information_list.append(mut_inf)
    lst.append(mutual_information_list)

lst=np.array(lst)
from matplotlib import pyplot as plt
heatmap = plt.pcolor(lst)
plt.colorbar()

#Have to check which interval the number fits into
#Don't need a full search the variables are in sequential order
    