"""
@author: Tor Kaufmann Gjerde
May 2021

A naive version of calulation of the Mandelbrot set.
Coded using Python’s standard library with focus 
on readability and ability to validate the implementation.

Update: This is the Cython version where type definition is used to speed up computations.
        should be built with the sepparate script:     setup_cython_naive_mandelbrot.py 
        This is done from terminal with the command:   setup_cython_naive_mandelbrot.py build_ext --inplace
        
        An already compiled and built version of this code is found in the same folder as: 
        cython_naive_mandelbrot.c 

"""

import matplotlib.pyplot as plt


cpdef create_matrix_real(double value_min, double value_max, int size):

    # Create matrix of defined size containing real values equally spaced
    # within predefined limits
    # "size" = size of the returned square matrix
    # Important: All columns are equal with same values 

    matrix = []
    cdef int row_count = size
    cdef int column_count = size

    # get the proper step-size that fits value_min to value_max
    cdef double data = (abs(value_min) + value_max) / (size-1)
    cdef int i
    cdef int j
    for i in range(row_count):
        row = []
       
        for j in range(column_count):
            if j == 0:
                row.append(value_min)
            else:
                row.append(value_min + (data * j))
        matrix.append(row)
    return matrix


cpdef create_matrix_imaginary(double value_min, double value_max, int size):
   
    # Create and return a matrix of defined size containing values equally spaced
    # within predefined limits: "value_min" and "value_max".
    # "size" is the size of the returned square matrix.
    # Important: All rows are equalwith same values 
    
    matrix = []
    cdef int row_count = size
    cdef int column_count = size

    # find the propper step-size from min to max
    cdef double data = (abs(value_min) + value_max) / (size - 1)
    cdef int i
    cdef int j
    for i in range(row_count):
        rowList = []
       
        for j in range(column_count):
            if i == 0:
                rowList.append(value_max)
            else:
                rowList.append(value_max - (data * i))
        matrix.append(rowList)
    return matrix


cpdef map_matrix(real_matrix, imaginary_matrix, int iterations, int threshold):
  
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
    
    cdef int size = len(real_matrix)
    if(len(real_matrix) != len(imaginary_matrix)):
        print("Error... real/imaginary matrix not equal in size")

    matrix = [] # initiate output matrix or python "list" 
    cdef int m
    cdef int n
    cdef int i
    cdef double complex c
    cdef double complex Z
    
    # fetch rows from Re and Im matrices for generating complex number
    for m in range(size):
        real_row = real_matrix[m]
        imag_row = imaginary_matrix[m]
        row = [] # initiate row list for 
        
        for n in range(size):
            c = complex(real_row[n], imag_row[n])
            Z = complex(0, 0)  # start value set to zero
            
            # do iterations on the complex value c
            for i in range(iterations):
                Z = Z**2 + c  # quadratic complex mapping

                if(abs(Z) > threshold):  # iteration "exploded"
                    # do mapping and stop current iteration
                    row.append(abs(Z)/iterations)
                    break
                # iterations did not "explode" therefore marked stable with a 1
                if(i == iterations - 1):
                    row.append(1)
                    
        # append completed row to list list matrix
        matrix.append(row)
    return matrix


cpdef plot_mandelbrot(map_matrix, xmin, xmax, ymin, ymax):

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
    
    

    