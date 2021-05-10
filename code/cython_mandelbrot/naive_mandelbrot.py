"""

@author: Tor Kaufmann Gjerde
March 2021

A naive version of calulation of the Mandelbrot set.
Written using Python’s standard library with focus 
on readability and ability to validate the implementation, 
 i.e. no Numpy functionality
 
"""

import time
import matplotlib.pyplot as plt


def create_matrix_real(value_min, value_max, size):

    # Create matrix of defined size containing real values equally spaced
    # within predefined limits
    # "size" = size of the returned square matrix
    # Important: All columns are equal with same values 
    
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
   
    # Create and return a matrix of defined size containing values equally spaced
    # within predefined limits: "value_min" and "value_max".
    # "size" is the size of the returned square matrix.
    # Important: All rows are equalwith same values 
    
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
    # Take the input matrices and generate a mapping matrix containing linear mapping 
    # of iterations done on the complex number in terms of mandelbrot computation. 
    # The complex number is constructed from the two matrices,
    # where one matrix contains the REAL part an the other matrix contains 
    # the IMAGINARY part. the matrices is indexed with the same index for a 
    # given complex number. 
    
    # INPUT:
    #    real_matrix:       Matrix containing all real components
    #    imaginary_matrix:  Matrix containing all imaginary components
    #    iterations:        Number of max iterations for mandelbrot computation
    #    threshold:         Threshold value for mandelbrot computation
    
    # OUTPUT:
    #    matrix:            Matrix with entries in the range [0, 1]
    
    
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
            c = complex(real_row[n], imag_row[n]) # get the complex number 

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


def plot_mandelbrot(map_matrix, xmin, xmax, ymin, ymax):

    # we can now plot the mandelbrot set using the matplot lib library 
    # using a matrix or python "list within a list" with mapped values as input.
    
    # INPUT: map_matrix -  matrix containing mandelbrot computations. 
    #            xmin   -  min value for Real component used in mandelbrot computations.
    #            xmax   -  max value for Real component used in mandelbrot computations.
    #            ymin   -  min value for Imaginary component used in mandelbrot computations.
    #            xmax   -  max value for Imaginary component used in mandelbrot computations. 
    
    
    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})
   
    fig.suptitle("MANDELBROT SET")
    im = ax.imshow(map_matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation="bicubic")

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()




if __name__ == '__main__':

    ITERATIONS = 200       # Number of iterations for mandelbrot computation
    THRESHOLD = 2          # Threshold for mandelbrot computation
    MATRIX_SIZE = 5000     # Square matrix dimension for values evaluated by
                           # mandelbrot algorithm 
    REAL_MAX = 1     
    REAL_MIN = -2
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start = time.time()
    real = create_matrix_real(REAL_MIN, REAL_MAX, MATRIX_SIZE)

    imag = create_matrix_imaginary(IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop  = time.time()

    print('Generated RE and IM matrices in:', stop-start, 'second(s)')
    print('Square matrix size:', MATRIX_SIZE)

    print("Mapping mandelbrot please wait...")

    start = time.time()
    map_matrix = map_matrix(real, imag, ITERATIONS, THRESHOLD)
    stop = time.time()

    print('Mapped mandelbrot set in:', stop-start, 'second(s)')

    flag = input("Plot mandelbrot set? [y]/[n]")

    if flag == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX  )
    else:
        print("Done cu to")
     
    