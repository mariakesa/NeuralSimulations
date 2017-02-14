# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:32:15 2015

@author: maria
"""

f=open('currentnmda.dat','r')
lines=f.readlines()
x=[]
y=[]
for i in range(2,len(lines)):
    l=lines[i].split(' ')
    x.append(float(l[0]))
    y.append(float(l[1]))
    
    
from matplotlib import pyplot as plt    

fig = plt.figure()
plt.plot(x,y)
fig.suptitle('Current time course, NMDA')
plt.xlabel('Time (ms)')
plt.ylabel('Current (nanoamperes)')
fig.savefig('currnmda.png')
