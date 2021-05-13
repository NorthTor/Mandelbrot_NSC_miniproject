"""
Author: Tor Kaufmann Gjerde
A numpy version of generating and plotting the Mandelbrot set
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import math
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
    :rtype complex_matrix: Numpy 2D array
    """

    real_array = np.linspace(real_min, real_max, size, dtype=np.float32)
    imag_array = np.linspace(imag_min, imag_max, size, dtype=np.float32)

    matrix = np.zeros((size, size), dtype=np.complex64)  # pre-allocating

    for m in range(size):
        for n in range(size):
            matrix[n, m] = complex(real_array[m], imag_array[n])

    return matrix


def map_matrix_mandelbrot(matrix, threshold, max_iter):
    """"
    Function that takes a square matrix containing complex values
    and does computation on each entry to check if its stable
    in terms of the mandelbrot set.
    Returns a 1:1 mapped matrix with entries in the range [0, 1].
    a value of 1 denotes a stable entry while a value below 1 signifies
    an unstable entry. The threshold variable can be visualized as the
    radius with starting point

    :param matrix: A 2D square matrix containing complex values
    :type matrix: numpy 2D array
    :param threshold: Threshold value for the mandelbrot algorithm
    :type threshold: float
    :param max_iter: Maximum iterations explored by the mandelbrot algorithm
    :type max_iter: int

    :return mapped_matrix: A matrix containing the mapped values
    :rtype matrix_mapped: numpy 2D array
    """

    size = len(matrix)
    # default "unstable"
    matrix_mapped = np.zeros((size, size), dtype=np.float32)  # pre-allocating

    for row in range(size):

        for column in range(size):
            c = matrix[row, column]  # fetch the complex number
            z = complex(0, 0)  # start value set to zero

            # do iterations on the complex value c
            for i in range(max_iter):
                z = z ** 2 + c  # quadratic complex mapping

                if abs(z) > threshold:  # iteration "exploded"
                    # do mapping and stop current iteration
                    matrix_mapped[row, column] = abs(z) / max_iter
                    break

                if i == max_iter - 1:  # iterations did not "explode", mark stable
                    matrix_mapped[row, column] = 1

    return matrix_mapped


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

    fig.suptitle('Mandelbrot set - numpy', fontweight='bold')

    im = ax.imshow(matrix, cmap='hot', extent=[x_min, x_max, y_min, y_max],
                   interpolation='bicubic')

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    MATRIX_SIZE = 5000  # Square matrix size (order)
    ITERATIONS = 200  # Number of iterations for mandelbrot algorithm
    THRESHOLD = 2  # Threshold (radius on the unit circle)

    REAL_MAX = 1  # Mesh set, maximum and minimum values
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    complex_matrix = create_complex_matrix(REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop = time.time()
    print('Generated matrix in:', stop - start, 'second(s)')
    print('Square matrix size (order):', MATRIX_SIZE)
    print('Mapping matrix... please wait')

    start = time.time()
    map_matrix = map_matrix_mandelbrot(complex_matrix, THRESHOLD, ITERATIONS)
    stop = time.time()
    print('Mapped generated matrix in:', stop - start, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        print("Saving.. please wait")
        file = h5py.File("numpy_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)

    print("Done cu mate!")
