# Author: To Kaufmann Gjerde

# A naive version of calculation of the Mandelbrot set.
# Write code using Python’s standard library with focus
# on readability and ability to validate the implementation,
# i.e. no Numpy func-tionality

import time
import matplotlib.pyplot as plt
import h5py

def create_matrix_real(value_min, value_max, size):
    # Create matrix of defined size containing real values equally spaced
    # within predefined limits
    # size = size of the returned square matrix

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
    # Create matrix of defined size containing imaginary values equally spaced
    # within predefined limits
    # size = size of the returned square matrix

    matrix = []
    row_count = size
    column_count = size

    # find the propper step-size from min to max
    data = (abs(value_min) + value_max) / (size - 1)

    for i in range(row_count):
        rowList = []
        for j in range(column_count):
            if i == 0:
                rowList.append(value_max)
            else:
                rowList.append(value_max - (data * i))
        matrix.append(rowList)
    return matrix


def map_matrix(real_matrix, imaginary_matrix, iterations, threshold):
    # Take the input matrices and generate a mapping matrix
    # containing linear mapping of iterations
    # INPUT:
    #    real_matrix:       Matrix containing all real components
    #    imaginary_matrix:  Matrix containing all imaginary components
    #    iterations:        Number of max iterations if not bound is hit
    #    threshold:         Threshold value
    #
    # OUTPUT:
    #    mapMatrix:         Matrix with entries in the range [0, 1]

    size = len(real_matrix)
    if (len(real_matrix) != len(imaginary_matrix)):
        print("Error... real/imaginary matrix not equal in size")

    matrix = []

    # fetch rows from Re and Im matrices for generating complex number
    for m in range(size):
        real_row = real_matrix[m]
        imag_row = imaginary_matrix[m]
        row = []

        for n in range(size):
            c = complex(real_row[n], imag_row[n])

            Z = complex(0, 0)  # start value set to zero
            flag = True
            # do iterations on the complex value c
            for i in range(iterations):
                Z = Z ** 2 + c  # quadratic complex mapping

                if (abs(Z) > threshold):  # iteration "exploded"
                    # do mapping and stop current iteration
                    row.append(abs(Z) / iterations)
                    flag = False
                    break
            # iterations did not "explode" therefore marked stable with a 1
            if (flag is True):
                row.append(1)

        # append completed row to mapMatrix
        matrix.append(row)
    return matrix


def plot_mandelbrot(matrix, xmin, xmax, ymin, ymax):
    # PLOT the mandelbrot from a mapped matrix with basis in a coordinate system
    # using the matplotlib package
    # x -> Real component of complex number
    # y -> Imaginary component of complex number

    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})

    fig.suptitle('Mandelbrot set', fontweight='bold')

    im = ax.imshow(matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation='bicubic')

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    MATRIX_SIZE = 1000  # Square matrix dimension
    ITERATIONS = 200    # Number of iterations for mandelbrot algorithm
    THRESHOLD = 2       # Threshold for mandelbrot algorithm

    REAL_MAX = 1
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    real = create_matrix_real(REAL_MIN, REAL_MAX, MATRIX_SIZE)

    imag = create_matrix_imaginary(IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop = time.time()
    print('Generated Matrix in:', stop - start, 'second(s)')
    print('Matrix dimension:', MATRIX_SIZE)

    start = time.time()
    map_matrix = map_matrix(real, imag, ITERATIONS, THRESHOLD)
    stop = time.time()
    print('Mapped generated matrix in:', stop - start, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        file = h5py.File("naive_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)
    else:
        print("Done cu mate!")
