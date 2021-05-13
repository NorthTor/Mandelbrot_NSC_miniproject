""""
Author: Tor Kaufmann Gjerde

A "naive" version of calculating the Mandelbrot set.
Code written using Python’s standard library with focus
on readability and ability to validate the implementation,
i.e. no Numpy functionality.
"""

import time
import matplotlib.pyplot as plt
import h5py


def create_matrix_real(value_min, value_max, size):
    """"
    Create square matrix of defined size containing values equally spaced
    within predefined limits. Same values along vertical axis

    :param value_min: Minimum value
    :type value_min: Double float (python)
    :param value_max: Maximum value
    :type value_max: Double float (python)
    :param size: Return square matrix size
    :type size: Int

    :return matrix: Matrix with values
    :rtype matrix: Python list of list
    """

    matrix = []
    row_count = size
    column_count = size

    # get the proper step-size that fits value_min to value_max
    data = (abs(value_min) + value_max) / (size - 1)

    for i in range(row_count):
        row = []
        for j in range(column_count):
            if j == 0:
                row.append(value_min)
            else:
                row.append(value_min + (data * j))
        matrix.append(row)
    return matrix


def create_matrix_imaginary(value_min, value_max, size):
    """"
    Create square matrix of defined size containing values equally spaced
    within predefined limits. Same values along horizontal axis

    :param value_min: Minimum value
    :type value_min: double float (python)
    :param value_max: Maximum value
    :type value_max: double float (python)
    :param size: Return square matrix size
    :type size: Int

    :return matrix: Matrix with values
    :rtype matrix: python list within list
    """
    matrix = []
    # find the proper step-size from min to max
    data = (abs(value_min) + value_max) / (size - 1)

    for i in range(size):
        row = []
        for j in range(size):
            if i == 0:
                row.append(value_max)
            else:
                row.append(value_max - (data * i))
        matrix.append(row)
    return matrix


def map_matrix_mandelbrot(real_matrix, imaginary_matrix, max_iter, threshold):
    """"
    Take the input matrices and generate a mapped matrix
    containing linear mapping using mandelbrot algorithm.

    :param real_matrix: Matrix containing real components
    :type real_matrix: Python list within list, double float
    :param imaginary_matrix: Matrix containing imaginary components
    :type imaginary_matrix: Python list within list, double float
    :param max_iter: Maximum iterations for mandelbrot algorithm
    :type max_iter: int
    :param threshold: threshold value for mandelbrot algorithm
    :type threshold: float

    :return matrix: Matrix containing mapped entries
    :rtype matrix: Python list within list, double float
    """

    size = len(real_matrix)
    if len(real_matrix) != len(imaginary_matrix):
        print("Error... real/imaginary matrix not equal size")

    matrix = []
    # fetch rows from Re and Im matrices for generating complex number
    for m in range(size):
        real_row = real_matrix[m]
        imag_row = imaginary_matrix[m]
        row = []

        for n in range(size):
            c = complex(real_row[n], imag_row[n])
            z = complex(0, 0)  # start value set to zero
            # do iterations on the complex value c
            for i in range(max_iter):
                z = z ** 2 + c  # quadratic complex mapping
                if abs(z) > threshold:  # iteration "exploded"
                    # do mapping and stop current iteration
                    row.append(abs(z) / max_iter)
                    break
                # iterations did not "explode" therefore marked stable with a 1
                if i == max_iter - 1:
                    row.append(1)
            # append completed row to output matrix
            matrix.append(row)
    return matrix


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

    fig.suptitle('Mandelbrot set - naive python', fontweight='bold')

    im = ax.imshow(matrix, cmap='hot', extent=[x_min, x_max, y_min, y_max],
                   interpolation='bicubic')

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    MATRIX_SIZE = 1000  # Square matrix size (order)
    ITERATIONS = 200  # Number of iterations for mandelbrot algorithm
    THRESHOLD = 2  # Threshold for mandelbrot algorithm

    REAL_MAX = 1
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    real = create_matrix_real(REAL_MIN, REAL_MAX, MATRIX_SIZE)
    imag = create_matrix_imaginary(IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop = time.time()
    print('Generated RE and IM matrices in:', stop - start, 'second(s)')
    print('Square matrix size (order):', MATRIX_SIZE)
    print('Mapping matrix... please wait')

    start = time.time()
    map_matrix = map_matrix_mandelbrot(real, imag, ITERATIONS, THRESHOLD)
    stop = time.time()
    print('Mapped generated matrix in:', stop - start, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        print("Saving... please wait")
        file = h5py.File("naive_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)

    print("Done cu mate!")
