# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:33:06 2015

@author: maria
"""

from brian import *

eqs_inhibitory='''
dv/dt=(-gl_i*(v-El_i)-g_ext_e*s_ext*(v-E_ampa)-g_gaba*s_gaba*(v-E_gaba))/Cm_i: volt
ds_ext/dt=-s_ext/t_ampa : 1
ds_gaba/dt=-s_gaba/t_gaba : 1
'''

eqs_excitatory='''
dv/dt=(-g_leak*(v-El_leak)-g_ext_i*s_ext*(v-E_ampa))/Cm_exc: volt
ds_ext/dt=-s_ext/t_ampa : 1
'''

nr_of_neurons=10
Cm_exc=0.5*nF
El_leak=-70*mV
g_leak=25*nS
V_th_exc=-50*mV
reset_exc=-60*mV
t_ref_exc=2*ms
Cm_inh=0.2*nF
gl_i=20*nS
El_i=-70*mV
V_th_inh=-50*mV
reset_inh=-60*mV
t_ref_inh=1*ms
t_ampa=2*ms
t_gaba=10*ms
fext=1.8*kHz
g_ext_e=3.1*nS
g_ext_i=2.38*nS
g_gaba=2.70*nS
E_gaba=-70*mV

exc_neurons = NeuronGroup(nr_of_neurons, eqs_excitatory,threshold=V_th_exc, reset=reset_exc,refractory=t_ref_exc)
inh_neurons = NeuronGroup(nr_of_neurons, eqs_inhibitory,threshold=V_th_exc, reset=reset_inh,refractory=t_ref_inh)

inputs = PoissonGroup(nr_of_neurons,fext)

input_connections = IdentityConnection(inputs,exc_neurons,'s_ext',weight=1.0)
input_conns = IdentityConnection(inputs,inh_neurons,'s_ext',weight=1.0)

inter_connectivity = Connection(exc_neurons,inh_neurons,'s_gaba',weight=1.0,sparsity=0.7)

M = SpikeMonitor(exc_neurons)
run(simulation_time)
spikes=M.spikes
raster_plot(M)