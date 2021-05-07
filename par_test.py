
# some usefull information: 
# https://stackoverflow.com/questions/8533318/multiprocessing-pool-when-to-use-apply-apply-async-or-map 


import time
import math 
import numpy as np
import multiprocessing 
import matplotlib.pyplot as plt
                  

def create_complex_matrix(real_min, real_max, imag_min, imag_max, size):
    """
    OUTPUT: matrix containing complex numbers
    """
    real_array = np.linspace(real_min, real_max, size, dtype=np.float32)
    imag_array = np.linspace(imag_min, imag_max, size, dtype=np.float32)
    
    complex_matrix = np.zeros((size, size), dtype=complex) # pre allocating 

    # Set up matrix with complex values
    for column in range(size):
        for row in range(size):
            complex_matrix[row, column] = complex(real_array[column], imag_array[row])

    return complex_matrix


def mandelbrot_computation(complex_nbr):
   iterations = 200
   threshold = 2
   # takes in an array with numpy complex numbers and does computation 
   # on all entries mapping them to the mandelbrot "range"
   #            1 = stable 
   # lower than 1 = more unstable 

   c = complex_nbr # fetch complex number from input data array 
   Z = complex(0, 0)  # start value set to zero
       
   # do iterations on the complex value c
   for i in range(iterations):
       
       Z = Z**2 + c  # quadratic complex mapping

       if(abs(Z) > threshold):  # iteration "exploded"
           # do mapping and stop current iteration
           mapped_entry = abs(Z)/(iterations)
           break
           
            # iterations did not "explode" therefore marked stable with a 1
       if(i == iterations-2):
            mapped_entry = 1
   
   return mapped_entry



def map_complex_matrix(complex_matrix, iterations, threshold, nbr_workers):
    
    """
    Function takes a matrix containing complex values and does computation 
    on each entry to check if its stable in terms of the mandelbrot set.
    multicore processing is used as a mean of parallelizing. 
    
    Returns a matrix with linear mapped antries in the range [0, 1] 
    a value of 1 denotes a stable entry while a value below 1 is deemed ustable 
    """
    #size = len(complex_matrix)
    #map_matrix = np.zeros((size, size))# preallocating 
    
    #for row in range(size):
        #complex_array = complex_matrix[row,:]
        
        # Take one row at the time and suply it to the workers. 
        # Workers executes the function: "mandelbrot_computation", where 
        # the argument is running throught all the rows of the complex_matrix.
        # Essetially paralleizing the computation of each row in the matrix 
        
    pool = multiprocessing.Pool(processes=(nbr_workers))
    map_array = pool.map(mandelbrot_computation, complex_matrix)
    
    pool.close() # prevent more tasks from being submitted to the pool 
    pool.join() # wait for worker processes to exit
        
        #map_matrix[row,:] = map_array
        
    return map_array

                
if __name__ == '__main__':

    
    ITERATIONS = 200         # Number of iterations
    THRESHOLD = 2            # Threshol (radius on the unit circle)
    MATRIX_SIZE = 1000       # Square matrix dimension
    WORKERS = 8             # Number of workers used for pool processing
    
    REAL_MATRIX_MAX = 1
    REAL_MATRIX_MIN = -2
    IMAG_MATRIX_MIN = -1.5
    IMAG_MATRIX_MAX = 1.5
    
    start = time.time()
    complex_matrix = create_complex_matrix(REAL_MATRIX_MIN, REAL_MATRIX_MAX, IMAG_MATRIX_MIN,
                                   IMAG_MATRIX_MAX, MATRIX_SIZE)
    stop = time.time()
    print('Generated matrix in:', stop-start, 'seconds')
    # Turn the generated matrix into 1D for mapping purposes
    complex_array = complex_matrix.flatten()
    
    start = time.time()
    map_array = map_complex_matrix(complex_array, ITERATIONS, THRESHOLD, WORKERS)
    stop = time.time()
    
    print('Mapped generated matrix in:', stop-start, 'seconds')
    print('Using', WORKERS, 'workers')
    # Turn the generated matrix  back into 2D for plotting purposes
    map_matrix =  np.reshape(map_array, (MATRIX_SIZE, MATRIX_SIZE)) 
    
    
    # we can PLOT the mandelbrot set
    xmin, xmax = REAL_MATRIX_MIN, REAL_MATRIX_MAX
    ymin, ymax = IMAG_MATRIX_MIN, IMAG_MATRIX_MAX

    fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})
    
    fig.suptitle("Mandelbrot set", fontweight='bold' )

    im = ax.imshow(map_matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation="bicubic")

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()
    



    
    
   