""""
Author: Tor Kaufmann Gjerde
Generating and plotting of the Mandelbrot set
A multicore parallel version using the python multiprocessing package
"""

import time
import math
import numpy as np
import multiprocessing
from itertools import repeat
import matplotlib.pyplot as plt
import h5py


def create_complex_matrix(real_min, real_max, imag_min, imag_max, size):
    """"
    Function that takes in a minimum and maximum value for the
    real and imaginary component of a complex number and returns
    a square matrix with complex entries of a specific size. The amount of
    steps between maximum and minimum values is linear with the size argument.
    The resulting matrix obtains the same real components in the vertical Y-direction
    and the same imaginary components in the horizontal X-direction.

    :param real_min: Minimum value of real component
    :type real_min: float32
    :param real_max: Maximum value of real component
    :type real_max: float32
    :param imag_min: Minimum value of imaginary component
    :type imag_min: float32
    :param imag_max: Maximum value of imaginary component
    :type imag_max: float 32
    :param size: Size of the output matrix
    :type size: float32

    :return complex_matrix: matrix with complex values
    :rtype complex_matrix: Numpy 2D array with float32 entries
    """

    real_array = np.linspace(real_min, real_max, size, dtype=np.float32)
    imag_array = np.linspace(imag_min, imag_max, size, dtype=np.float32)

    matrix = np.zeros((size, size), dtype=np.complex64)  # pre-allocating

    # Set up matrix with complex values
    for column in range(size):
        for row in range(size):
            matrix[row, column] = complex(real_array[column], imag_array[row])

    return matrix


def mandelbrot_computation(complex_nbr, max_iter, threshold):
    """"
    Function that takes in a complex numbers and does computation
    on it mapping it to the mandelbrot "range"
    where:
      mapped  value = 1 : Complex number stable and part of mandelbrot set
      mapped value < 1  : Degree of unstable and not part of mandelbrot set

    :param complex_nbr: A complex number
    :type complex_nbr: Numpy.complex64
    :param max_iter: Maximum iterations for mandelbrot computation
    :type max_iter: Int
    :param threshold: threshold for mandelbrot computation
    :type threshold: float

    :return mapped_value: Mapped value in the range [0 1]
    :rtype mapped_value: numpy.float64
    """

    mapped_value = np.float64(0) # default unstable
    c = complex_nbr  # fetch complex number from input data array
    z = complex(0, 0)  # start value set to zero

    # do iterations on the complex value c
    for i in range(max_iter):
        z = z ** 2 + c  # quadratic complex mapping
        if abs(z) > threshold:  # iteration "exploded"
            # do mapping and stop current iteration
            mapped_value = abs(z) / max_iter
            break
            # iterations did not "explode" therefore marked stable with a 1
        if i == max_iter - 1:
            mapped_value = 1

    return mapped_value


def map_mandelbrot_multicore(array, iterations, threshold, nbr_workers):
    """
    Function takes an array containing complex values and does computation
    on each entry to check if its stable in terms of the mandelbrot set.
    multicore processing is used as a mean of parallelizing.

    # Returns an array with linear mapped entries in the range [0, 1]
    # a value of 1 denotes a stable entry while a value below 1 is deemed unstable

    # Takes one row  at the time and supply it to the workers.
    # Workers executes the function: "mandelbrot_computation", where
    # the argument is running through all the rows of the complex_matrix.
    # Essentially parallelizing the computation of each row in the final matrix

    :param array: Array containing complex values
    :type array: numpy 1D array
    :param iterations: Maximum iterations for the mandelbrot algorithm
    :type iterations: Int
    :param threshold: Threshold for the mandelbrot algorithm
    :type threshold: float
    :param nbr_workers: Number of workers used in parallel
    :type nbr_workers: Int

    :return map_array: Array with mapped values
    :rtype map_array: Numpy 1D array
    """

    pool = multiprocessing.Pool(processes=nbr_workers)
    map_array = pool.starmap(mandelbrot_computation, zip(array, repeat(iterations), repeat(threshold)))

    pool.close()  # prevent more tasks from being submitted to the pool
    pool.join()  # wait for worker processes to exit

    return map_array


def plot_mandelbrot(matrix, x_min, x_max, y_min, y_max):
    """
    PLOT the mandelbrot from a mapped matrix with basis in a coordinate system
    using the matplotlib package. The maximum/minimum values should agree with
    maximum/minimum values for the components of the complex numbers used in
    generating the mapped matrix.

    :param matrix: Mapped matrix to be plotted
    :type matrix: Numpy 2D array
    :param x_min: Minimum value for horizontal axis, x-axis
    :type: x_min: float32
    :param x_max: Maximum value for horizontal axis, x-axis
    :type x_max: float32
    :param y_min: minimum value for vertical axis, y-axis
    :type y_min: float32
    :param y_max: Maximum value for vertical axis, y-axis
    :type y_max: float32
    """

    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})

    fig.suptitle('Mandelbrot set - multicore python', fontweight='bold')

    im = ax.imshow(matrix, cmap='hot', extent=[x_min, x_max, y_min, y_max],
                   interpolation='bicubic')

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    MATRIX_SIZE = 1000  # Square matrix size (order)
    ITERATIONS = 200  # Number of iterations
    THRESHOLD = 2  # Threshold (radius on the unit circle)

    WORKERS = 2  # Number of workers used for pool processing

    REAL_MAX = 1
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    complex_matrix = create_complex_matrix(REAL_MIN, REAL_MAX,
                                           IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop = time.time()

    print('Generated Matrix in:', stop - start, 'second(s)')
    print('Square matrix size:', MATRIX_SIZE)
    print('Mapping matrix using:', WORKERS, 'workers')

    # turn the generated matrix into 1D for mapping purposes
    complex_array = complex_matrix.flatten()

    start = time.time()
    map_array = map_mandelbrot_multicore(complex_array, ITERATIONS, THRESHOLD, WORKERS)
    stop = time.time()

    # turn the generated matrix  back into 2D for plotting purposes
    map_matrix = np.reshape(map_array, (MATRIX_SIZE, MATRIX_SIZE))

    print('Mapped generated matrix in:', stop - start, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        print("Saving.. please wait")
        file = h5py.File("multicore_parallel_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)

    print("Done cu mate!")
