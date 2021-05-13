"""
Author: Tor Kaufmann Gjerde
May 2021

Following code run the cython version of mandelbrot computation and plotting

Other dependencies:
  cython_naive_mandelbrot.pyx
  setup_cython_naive_mandelbrot.py
  cython_naive_mandelbrot.c

  Check out the file "cython_naive_mandelbrot.pyx"  for how to compile
"""

import cython_naive_mandelbrot
import time
import matplotlib.pyplot as plt
import h5py


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

    fig.suptitle('Mandelbrot set - naive Cython', fontweight='bold')

    im = ax.imshow(matrix, cmap='hot', extent=[x_min, x_max, y_min, y_max],
                   interpolation='bicubic')

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    MATRIX_SIZE = 5000  # Square matrix size (order)
    ITERATIONS = 200  # Number of mandelbrot algorithm iterations
    THRESHOLD = 2  # Threshold for mandelbrot algorithm

    REAL_MAX = 1
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    real = cython_naive_mandelbrot.create_matrix_real(REAL_MIN, REAL_MAX, MATRIX_SIZE)
    imag = cython_naive_mandelbrot.create_matrix_imaginary(IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop = time.time()

    print('Generated RE and IM matrices in:', stop - start, 'second(s)')
    print('Square matrix size:', MATRIX_SIZE)

    print("Mapping mandelbrot please wait...")

    start = time.time()
    map_matrix = cython_naive_mandelbrot.map_matrix_mandelbrot(real, imag, ITERATIONS, THRESHOLD)
    stop = time.time()

    print('Mapped mandelbrot set in:', stop - start, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        file = h5py.File("cython_naive_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)

    print('Done cu to')
