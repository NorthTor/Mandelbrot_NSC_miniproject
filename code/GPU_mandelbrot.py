#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tor Kaufmann Gjerde

A GPU accelerated using PyOpenCl
version of generation and plotting of the Mandelbrot set

"""
import time
import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl


def create_real_and_imag_matrices(real_min, real_max, imag_min, imag_max, size):
# Function that returns two square matrices with dimension "size"
# OUTPUT: real_matrix, imag_matrix
    

    real_array = np.linspace(real_min, real_max, size, dtype=np.float32) # set up a vector with linear spacing netween min and max values 
    imag_array = np.linspace(imag_max, imag_min, size, dtype=np.float32) # set up a vector with linear spacing netween min and max values 
    real_matrix = np.zeros((size, size), dtype=np.float32) # pre allocating output vector 
    imag_matrix = np.zeros((size, size), dtype=np.float32) # pre allocating output vector 

    # Set up matrix with complex values 
    for n in range(size):
            real_matrix[n,:] = real_array  #insert into output vector
            imag_matrix[:,n] = imag_array  #insert into output vector
            
    return real_matrix, imag_matrix


def create_map_matrix_GPU(real_matrix, imag_matrix):
    # Create the context (containing platform and device information).
    context = cl.create_some_context()
    # Kernel execution, synchronization, and memory transfer 
    # opperations are submitted through the command que
    # each command queue points to a single device within a context. Create command que: 
    cmd_queue = cl.CommandQueue(context)

    real_matrix_host = real_matrix # matrix containing real parts 
    imag_matrix_host = imag_matrix # matrix containing imaginary parts 

    # Create empty matrix to hold the resulting mapped matrix
    result_host = np.empty((SIZE,SIZE)).astype(np.float32)

    # Create a device side read-only memory buffer and copy the data from "hostbuf" into it.
    # Other possible mem_flags values at:
    # https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
    mf = cl.mem_flags
    real_matrix_device = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=real_matrix_host) 
    imag_matrix_device = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=imag_matrix_host)

    result_device = cl.Buffer(context, mf.READ_WRITE, result_host.nbytes)

    # Source of the kernel itself.
    kernel_source = open("GPU_mandelbrot_kernel.cl").read()

    # Create a new program from the kernel and build the source.
    prog = cl.Program(context, kernel_source).build()

    # Execute the kernel in the program. Parameters are:
    # 
    #            Command queue            Work group size               Kernel param 1
    #                ↓       Global grid size   ↓   Kernel param 0           ↓          Kernel param 2
    #                ↓               ↓          ↓       ↓                    ↓                  ↓
    prog.mandelbrot(cmd_queue, (SIZE*SIZE,), None, real_matrix_device, imag_matrix_device, result_device, np.int32(ITERATIONS), np.int32(THRESHOLD))
        
    # Copy the result back from device to host.
    cl.enqueue_copy(cmd_queue, result_host, result_device)

    return(result_host)


def plot_mandelbrot_set(matrix):
    # we can PLOT the mandelbrot set
    xmin, xmax = REAL_MATRIX_MIN, REAL_MATRIX_MAX
    ymin, ymax = IMAG_MATRIX_MIN, IMAG_MATRIX_MAX

    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})
    
    fig.suptitle("Mandelbrot set", fontweight='bold' )

    im = ax.imshow(matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation="bicubic")

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    
    REAL_MATRIX_MIN = -2
    REAL_MATRIX_MAX = 1
    IMAG_MATRIX_MIN = -1.5
    IMAG_MATRIX_MAX = 1.5

    SIZE = 10000       # Matrix dimension 
    ITERATIONS = 1000  # Iterations for mandelbrot kernel
    THRESHOLD = 2      # Threshold used in mandelbrot kernel
    
    start1 = time.time()
    real_matrix, imag_matrix = create_real_and_imag_matrices(REAL_MATRIX_MIN, REAL_MATRIX_MAX, 
                                                             IMAG_MATRIX_MIN, IMAG_MATRIX_MAX, 
                                                             SIZE)
    stop1 = time.time()
    
    start2 = time.time()
    map_matrix = create_map_matrix_GPU(real_matrix, imag_matrix )
    stop2 = time.time()
    
    print('Generated matrix in:', stop1-start1, 'second(s)')
    print('Mapped generated matrix in:', stop2-start2, 'second(s)')
    
    plot_mandelbrot_set(map_matrix)




