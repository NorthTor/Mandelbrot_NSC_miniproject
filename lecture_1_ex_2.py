#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:15:30 2021

@author: torkaufmanngjerde
"""

# Exercise 1.2: Cache impact on different problem sizes

# Matrix-vector product b = Ax,
# A is an M x M matrix
# x is a M x 1 vector

import numpy as np
import time
import matplotlib.pyplot as plt

low = 0
high = 10
max_k = 13
nbr_exe = 100 # number of individual executions to average over 

# initialize arrays
time_array = np.array([0] * (nbr_exe), dtype='f')         # array holding timings
average_time_array = np.array([0] * (max_k), dtype='f')   # array holding averaged timings
FLOPS = np.array([0] * (max_k), dtype='i')                # array holdig the amount of FLOPS

startX = time.time() 

for i in range(0,max_k):
    M = 2**(i+1)
    for index in range(0,nbr_exe):
        # Generate A = M x M matrix with random int values 
        A = np.random.randint(low,high, size=(M, M))
        # Generate vector x of size M
        x = np.random.randint(low,high, size=(M, 1))
    
        start = time.time()  # start timer 
        b = np.dot(A,x)      # do matrix vector product
        end = time.time()    # end timer
        
        value = end - start  # time taken to do matrix vector product
        time_array[index] = value # store value
    
    average_time_array[i] = np.average(time_array) # average the time 




nbr_arit_op = np.array([0] * (max_k)) # Initialize array to hold amount of operations 
                                                                       
endX = time.time()  
print("Seconds used executing loop:", endX - startX)   
                             
for i in range(0,max_k):
   M = i+2
   nbr_arit_op[i] = (2* M**2) - M # Array holding the number of arithmetic 
                                  # operations as k increase (2^k  where k = 1,2,..13)
   
   FLOPS[i] = (nbr_arit_op[i] / average_time_array[i])

print(FLOPS)
plt.plot(nbr_arit_op, FLOPS)
plt.ylabel('FLOPS')
plt.xlabel('Number of arithmetic operations')
plt.grid(True)
plt.show()
print("done")



