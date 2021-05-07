#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:22:26 2021

@author: torkaufmanngjerde

A naive version of calulation of the Mandelbrot set.
Write code using Python’s standard library with focus 
on readability and ability to validate the implementation, 
 i.e. no Numpy func- tionality
 
"""

import time
import matplotlib.pyplot as plt


def create_matrix_real(value_min, value_max, size):
    """
    create matrix of defined size containing real values equally spaced
    within predefined limits
    size = size of the returned square matrix
    """
    matrix = []
    row_count = size
    column_count = size

    # get the proper step-size that fits value_min to value_max
    data = (abs(value_min) + value_max) / (size-1)

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
    """
    create matrix of defined size containing imaginary values equally spaced
    within predefined limits
    size = size of the returned square matrix
    """
    
    matrix = []
    row_count = size
    column_count = size

    # find the propper step-size from min to max
    data = (abs(value_min) + value_max) / (size-1)

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
    """
    Take the input matrises and generate a mapping matrix
    containing linear mapping of iterations
    INPUT:
        real_matrix:       Matrix containing all real components
        imaginary_matrix:  Matrix containing all imaginary components
        iterations:        Number of max iterations if not bound is hit
        threshold:         Threshold value
    
    OUTPUT:
        mapMatrix:         Matrix with entries in the range [0, 1]
    """
    
    size = len(real_matrix)
    if(len(real_matrix) != len(imaginary_matrix)):
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
                Z = Z**2 + c  # quadratic complex mapping

                if(abs(Z) > threshold):  # iteration "exploded"
                    # do mapping and stop current iteration
                    row.append(abs(Z)/iterations)
                    flag = False
                    break
            # iterations did not "explode" therefore marked stable with a 1
            if(flag is True):
                row.append(1)

        # append completed row to mapMatrix
        matrix.append(row)
    return matrix


if __name__ == '__main__':

    ITERATIONS = 100    # Number of iterations
    THRESHOLD = 2       # Threshol
    MATRIX_SIZE = 1000   # Square matrix dimension
    REAL_MATRIX_MAX = 1
    REAL_MATRIX_MIN = -2
    IMAG_MATRIX_MIN = -1.5
    IMAG_MATRIX_MAX = 1.5

    start = time.time()
    real = create_matrix_real(REAL_MATRIX_MIN,
                              REAL_MATRIX_MAX, MATRIX_SIZE)

    imag = create_matrix_imaginary(IMAG_MATRIX_MIN,
                                   IMAG_MATRIX_MAX, MATRIX_SIZE)
    stop = time.time()
    print(stop-start)

    start = time.time()
    map_matrix = map_matrix(real, imag, ITERATIONS, THRESHOLD)
    stop = time.time()
    print(stop-start)

    # we can plot the mandelbrot set using data from the "naive method"
    xmin, xmax = REAL_MATRIX_MIN, REAL_MATRIX_MAX
    ymin, ymax = IMAG_MATRIX_MIN, IMAG_MATRIX_MAX

    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})
    fig.suptitle("MANDELBROT SET")

    im = ax.imshow(map_matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation="bicubic")

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()
    