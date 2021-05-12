# Author: Tor Kaufmann Gjerde
# A numpy version of plotting of the Mandelbrot set

import time
import numpy as np
import matplotlib.pyplot as plt
import math 
import h5py

def create_complex_matrix(real_min, real_max, imag_min, imag_max, size):
    # OUTPUT: matrix containing complex numbers

    real_array = np.linspace(real_min, real_max, size, dtype=np.float32)
    imag_array = np.linspace(imag_min, imag_max, size, dtype=np.float32)
    
    complex_matrix = np.zeros((size, size), dtype=complex) # pre allocating 

    # set up matrix with complex values
    for column in range(size):
        for row in range(size):
            complex_matrix[row, column] = complex(real_array[column], imag_array[row])

    return complex_matrix


def map_matrix(complex_matrix, threshold, iterations):

    # Function takes a matrix containing complex values and does computation
    # on each entry to check if its stable in terms of the mandelbrot set
    #
    # Returns a matrix with linear mapped entries in the range [0, 1]
    # a value of 1 denotes a stable entry while a value below 1 is deemed unstable

    size = len(complex_matrix)
    map_matrix = np.zeros((size, size),dtype=np.float32)  # pre-allocating
    
    for row in range(size):
        
        for column in range(size):
            c = complex_matrix[row,column] # fetch the complex number 
            Z = complex(0, 0)  # start value set to zero
            
            # do iterations on the complex value c
            for i in range(iterations):
                Z = Z**2 + c  # quadratic complex mapping

                if(abs(Z) > threshold):  # iteration "exploded"
                    # do mapping and stop current iteration
                    map_matrix[row, column] = abs(Z)/(iterations)
                    break
                
                if(i == iterations - 1): # iterations did not "explode", mark stable with a 1
                    map_matrix[row, column]= 1
                
    return map_matrix


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

    MATRIX_SIZE = 1000  # Square matrix dimension
    ITERATIONS = 200    # Number of iterations
    THRESHOLD = 2       # Threshold (radius on the unit circle)
    
    REAL_MAX = 1        # Mesh set, maximum and minimum values
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    complex_matrix = create_complex_matrix(REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop = time.time()
    
    print('Generated matrix in:', stop-start, 'second(s)')
    
    start = time.time()
    map_matrix = map_matrix(complex_matrix, THRESHOLD, ITERATIONS)
    stop = time.time()
    print('Mapped generated matrix in:', stop-start, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        file = h5py.File("numpy_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)
    else:
        print("Done cu mate!")