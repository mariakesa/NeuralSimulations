# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:02:02 2015

@author: maria
"""

from brian import *
'''
This is unfinished work.
'''
#########
#Number of Neurons
##########
nr_of_inhibitory_neurons=512
nr_of_excitatory_neurons=2048
#################
#Simulation clock
#################
simulation_clock=Clock(dt=0.02*ms)
############################
#Inhibitory Neurons
############################
inh_eqs='''
dv/dt=(-gl_i*(v-El_i)-g_ext_i*s_ext*(v-E_ampa)-Gii*s_gaba*(v-E_gaba)-Gei*s_tot*(v-E_nmda)/(1+b*exp(-a*v/(1*mV))))/Cm_i: volt
ds_ext/dt=-s_ext/t_ampa : 1
ds_gaba/dt=-s_gaba/t_gaba :1
ds_nmda/dt = -s_nmda/t_nmda+alpha*x*(1-s_nmda) : 1
dx/dt = -x/t_x :1
s_tot :1
'''

Cm_i=0.2*nF
gl_i=20*nS
El_i=-70*mV
inh_th=-50*mV
inh_reset=-60*mV
nmda_reset='''
v=inh_reset
x=1*1
'''
inh_ref=1*ms
g_ext_i=2.38*nS
t_ampa=2*ms
t_gaba=10*ms
E_gaba=-70*mV
E_ampa=0*mV
fext=1.8*kHz
Gii=1.024*nS
Gei=0.292*nS

b=1.0/3.57
a=0.062

######################################
#Excitatory neurons
######################################

exc_eqs='''
dv/dt=(-g_l_exc*(v-E_l_exc)-g_ext_e*s_ext*(v-E_ampa)-Gie*s_gaba*(v-E_gaba)-Gee*s_tot*(v-E_nmda)/(1+b*exp(-a*v/(1*mV)))+I_e)/C_exc: volt
ds_ext/dt=-s_ext/t_ampa : 1
ds_gaba/dt=-s_gaba/t_gaba :1
ds_nmda/dt = -s_nmda/t_nmda+alpha*x*(1-s_nmda) : 1
dx/dt = -x/t_x :1
s_tot :1
I_e : pA
'''

#Excitatory Neurons
C_exc = 0.5*nF
g_l_exc = 25*nS
E_l_exc = -70*mV
V_th_exc = -50*mV
V_reset_exc  = -60*mV
ref_exc  = 2*ms

#AMPA constants
E_ampa=0*mV
t_ampa=2*ms
g_ext_e=3.1*nS

#GABA constants
Gie=1.366*nS
E_gaba=-70*mV
t_gaba=10*ms

#NMDA constants
Gee=0.381*nS
t_nmda=100*ms
t_x=2*ms
b=1.0/3.57
a=0.062
alpha= 2.0*kHz
E_nmda=0*mV
reset='''
v=V_reset_exc
x+=1*1
'''
    

####################
#Defining neuron groups
####################
inhibitory_neurons=NeuronGroup(nr_of_inhibitory_neurons,inh_eqs,threshold=inh_th,reset=nmda_reset,refractory=inh_ref, clock=simulation_clock, order=2)

exc_neurons = NeuronGroup(nr_of_excitatory_neurons,exc_eqs,threshold=V_th_exc,reset=reset,refractory=ref_exc, clock=simulation_clock, order=2)
##############################
#Interconnectivity#
##############################
#IE,EI,II,EE+Poisson inputs through GABA

input_exc_Poisson = PoissonGroup(nr_of_excitatory_neurons,fext, clock=simulation_clock)
input_conn  = IdentityConnection(input_exc_Poisson,exc_neurons,'s_ext')

input_signal=PoissonGroup(nr_of_inhibitory_neurons,fext, clock=simulation_clock)
stimulus_inhibitory_connection=IdentityConnection(input_signal,inhibitory_neurons,'s_ext',weight=1.0)

conn_II=Connection(inhibitory_neurons, inhibitory_neurons, 's_gaba',weight=1.0)
conn_IE= Connection(inhibitory_neurons,exc_neurons,'s_gaba',weight=1.0)
conn_EE = Connection(exc_neurons,exc_neurons,'s_tot',weight=1.0)
conn_EI=Connection(exc_neurons,inhibitory_neurons,'s_tot',weight=1.0)

#########################
#Pre-computation of the weights
#########################
#THIS PART SHOULD BE OKAY
from scipy.special import erf
from numpy.fft import rfft, irfft

Jp_ee=1.62
sigma_ee=14.4

tmp =  sqrt(2*pi)*sigma_ee*erf(360.*0.5/sqrt(2.)/sigma_ee)/360
Jm_ee=(1.-Jp_ee*tmp)/(1.-tmp)
weight= lambda i: (Jm_ee+(Jp_ee-Jm_ee)* exp(-0.5*(360.*min(i,nr_of_excitatory_neurons-i)/nr_of_excitatory_neurons)**2/sigma_ee**2))

weight_e=zeros(nr_of_excitatory_neurons)
for i in xrange(nr_of_excitatory_neurons):    
    weight_e[i]=weight(i)
    
fweight=rfft(weight_e)


###########################
NMDA_profile=[]
NMDA_profile_inh=[]
@network_operation(clock=simulation_clock, when='start')
def update_nmda(clock=simulation_clock):
    s_NMDA1=irfft(rfft(exc_neurons.s_nmda)*fweight)
    NMDA_profile.append(s_NMDA1[1000]) 
    s_NMDA2=exc_neurons.s_nmda.sum() 
    NMDA_profile_inh.append(s_NMDA2)
    exc_neurons.s_tot=s_NMDA1
    inhibitory_neurons.s_tot=s_NMDA2

###########################
#Input stimulation
###########################

i_cue_ang=180
i_cue_amp=200
i_cue_width=10
def circ_distance(deltaTheta):
    if (deltaTheta>0):
        return min(deltaTheta,360-deltaTheta)
    else:
        return max(deltaTheta,deltaTheta-360)
        
currents = lambda i,j : i_cue_amp*exp(-0.5*circ_distance
((i-j)*360./nr_of_excitatory_neurons)**2/i_cue_width**2)
current_e = zeros(nr_of_excitatory_neurons)
j= i_cue_ang*nr_of_excitatory_neurons/360.

for i in xrange(nr_of_excitatory_neurons):
    current_e[i]=currents(i,j)
   
current_clock=Clock(dt=50*ms)
tc_start=1*second
tc_stop=1.25*second    
@network_operation(current_clock, when='start')
def update_currents(current_clock):
    if (current_clock.t>tc_start and current_clock.t<tc_stop):
        exc_neurons.I_e=current_e
        
    else:
        exc_neurons.I_e=0
    
simulation_time=5*second

M = SpikeMonitor(inhibitory_neurons)
Z = SpikeMonitor(exc_neurons)
R= SpikeMonitor(input_exc_Poisson)
O= SpikeMonitor(input_signal)
run(simulation_time)
spikes=M.spikes
#raster_plot(Z)

plot(NMDA_profile_inh)