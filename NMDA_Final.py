# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 00:58:53 2015

@author: maria
"""
from brian import *


################################################
#Simulations with NMDA and AMPA synapses
################################################
eqs_exc='''
dv/dt=(-gl_e*(v-El_e)-g_ext_e*s_ext*(v-E_ampa)-Gee*s_tot*(v-E_nmda)/(1+b*exp(-1/a*v)))/Cm_e : volt
ds_ext/dt=-s_ext/t_ampa : 1
ds_nmda/dt=-s_nmda/t_nmda+alpha*x*(1-s_nmda): 1
dx/dt=-x/t_x : 1
s_tot : 1
'''
simulation_time=0.5*ms
nr_of_neurons=10
Cm_e=0.5*nF
gl_e=25*nS
El_e=-70*mV
a=1/0.062*mV
b=3.57
t_x=2*ms
t_nmda=100*ms
alpha=0.5*kHz
Gee=0.381*nS
V_th_exc=-50*mV
reset_exc='''
v=Vr_e
x+=1*1
'''
E_nmda=-65*mV
Vr_e=-60*mV
t_ref_exc=2*ms
simulationclock=Clock(dt=0.02*ms)
exc_neurons = NeuronGroup(nr_of_neurons, eqs_exc,threshold=V_th_exc, reset=reset_exc,refractory=t_ref_exc, clock=simulation_clock)

@network_operation(clock=simulation_clock, when='start')
def update_nmda(clock=simulation_clock):
    s_NMDA=exc_neurons.s_nmda.sum()
    exc_neurons.s_tot=s_NMDA
    

inputs = PoissonGroup(nr_of_neurons,fext,clock=simulation_clock)
inter_connectivity = Connection(exc_neurons,exc_neurons,'s_tot',weight=1.0,sparsity=0.7)
input_connections = IdentityConnection(inputs,exc_neurons,'s_ext',weight=1.0)

M = SpikeMonitor(exc_neurons)
run(simulation_time)
spikes2=M.spikes
#raster_plot(M)

##############################################
#Simulations with only AMPA synapses
##############################################
import pyentropy
import matplotlib.pyplot as plt

eqs_i='''
dv/dt=(-gl_i*(v-El_i)-g_ext_i*s_ext*(v-E_ampa))/Cm_i: volt
ds_ext/dt=-s_ext/t_ampa : 1
'''

defaultclock.dt = 0.02*ms
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
neuron_connection = Connection(neurons,neurons,'s_ext',weight=1.0,sparseness=0.7)

#inputs=PoissonGroup(10,rates=linspace(0*kHz,5*kHz,10))

inputs = PoissonGroup(nr_of_neurons,fext)

input_connections = IdentityConnection(inputs,neurons,'s_ext',weight=1.0)

M = SpikeMonitor(neurons)
run(simulation_time)
spikes1=M.spikes


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
spike_list1=discretize_spikes(nr_of_bins,simulation_time,spikes1,nr_of_neurons)
spike_list1=discretize_spikes(nr_of_bins,simulation_time,spikes2,nr_of_neurons)

    
def calculate_mutual_information(X,Y,n):
    s=pyentropy.systems.DiscreteSystem(X,(1,n), Y,(1,n))
    s.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
    return s.I()

lst=[]
for i in range(0,nr_of_neurons):
    mutual_information_list=[]
    for j in range(0,nr_of_neurons):
        mut_inf = calculate_mutual_information(spike_list1[i],spike_list2[j],nr_of_bins)           
        mutual_information_list.append(mut_inf)
    lst.append(mutual_information_list)

lst=np.array(lst)
from matplotlib import pyplot as plt
heatmap = plt.pcolor(lst)
plt.colorbar()