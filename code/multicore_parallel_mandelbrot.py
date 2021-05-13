# Author: Tor Kaufmann Gjerde
# Generating and plotting of the Mandelbrot set
# A multicore parallel version using the python multiprocessing package

import time
import math
import numpy as np
import multiprocessing
from itertools import repeat
import matplotlib.pyplot as plt
import h5py

def create_complex_matrix(real_min, real_max, imag_min, imag_max, size):
    # A function that sets up a matrix with complex entries
    # INPUT: range for imaginary and real components of complex number
    # OUTPUT: matrix containing complex numbers

    real_array = np.linspace(real_min, real_max, size, dtype=np.float32)
    imag_array = np.linspace(imag_min, imag_max, size, dtype=np.float32)

    complex_matrix = np.zeros((size, size), dtype=complex)  # pre-allocating

    # Set up matrix with complex values
    for column in range(size):
        for row in range(size):
            complex_matrix[row, column] = complex(real_array[column], imag_array[row])

    return complex_matrix


def mandelbrot_computation(complex_nbr, iterations, threshold):
    # takes in an array with complex numbers and does computation
    # on all entries mapping them to the mandelbrot "range"
    #            1 = stable 
    # lower than 1 = more unstable 
    # INPUT: a complex number,
    #        number of iterations for mandelbrot algorithm
    #        threshold value for mandelbrot algorithm

    c = complex_nbr  # fetch complex number from input data array
    Z = complex(0, 0)  # start value set to zero

    # do iterations on the complex value c
    for i in range(iterations):

        Z = Z ** 2 + c  # quadratic complex mapping

        if abs(Z) > threshold:  # iteration "exploded"
            # do mapping and stop current iteration
            mapped_entry = abs(Z) / iterations
            break

            # iterations did not "explode" therefore marked stable with a 1
        if i == iterations - 2:
            mapped_entry = 1

    return mapped_entry


def map_array_multicore(array, iterations, threshold, nbr_workers):
    # Function takes a matrix containing complex values and does computation
    # on each entry to check if its stable in terms of the mandelbrot set.
    # multicore processing is used as a mean of parallelizing.

    # Returns a matrix with linear mapped entries in the range [0, 1]
    # a value of 1 denotes a stable entry while a value below 1 is deemed unstable

    # Take one row at the time and supply it to the workers.
    # Workers executes the function: "mandelbrot_computation", where
    # the argument is running through all the rows of the complex_matrix.
    # Essentially parallelizing the computation of each row in the matrix

    pool = multiprocessing.Pool(processes=nbr_workers)
    map_array = pool.starmap(mandelbrot_computation, zip(array, repeat(iterations), repeat(threshold)))

    pool.close() # prevent more tasks from being submitted to the pool
    pool.join()  # wait for worker processes to exit

    # map_matrix[row,:] = map_array

    return map_array


def plot_mandelbrot(matrix, xmin, xmax, ymin, ymax):
    # PLOT the mandelbrot from a mapped matrix with basis in a coordinate system
    # using the matplotlib package
    # x -> Real component of complex number
    # y -> Imaginary component of complex number

    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})

    fig.suptitle("Mandelbrot set", fontweight='bold')

    im = ax.imshow(matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation="bicubic")

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    MATRIX_SIZE = 2000  # Square matrix dimension
    ITERATIONS = 200    # Number of iterations
    THRESHOLD = 2       # Threshold (radius on the unit circle)

    WORKERS = 2         # Number of workers used for pool processing

    REAL_MAX = 1
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    complex_matrix = create_complex_matrix(REAL_MIN, REAL_MAX, IMAG_MIN,
                                           IMAG_MAX, MATRIX_SIZE)
    stop = time.time()
    print('Generated Matrix in:', stop - start, 'second(s)')
    print('Matrix dimension:', MATRIX_SIZE)
    # turn the generated matrix into 1D for mapping purposes
    complex_array = complex_matrix.flatten()
    print(type(complex_array))
    start = time.time()
    map_array = map_array_multicore(complex_array, ITERATIONS, THRESHOLD, WORKERS)
    stop = time.time()
    # turn the generated matrix  back into 2D for plotting purposes
    map_matrix = np.reshape(map_array, (MATRIX_SIZE, MATRIX_SIZE))
    print('Mapped generated matrix in:', stop-start, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        file = h5py.File("multicore_parallel_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)
    else:
        print("Done cu mate!")
