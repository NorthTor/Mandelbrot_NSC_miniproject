
# Tor Kaufmann Gjerde 

# dask_parallel_mandelbrot.py 
# A dask implementation for computing and plotting the mandelbrot set 


import time
import numpy as np
import matplotlib.pyplot as plt

from dask.distributed import Client, wait
import dask.delayed as delay
import webbrowser

                  

def create_complex_matrix(real_min, real_max, imag_min, imag_max, size):
    # Create a square matrix with complex entries for mandelbrot computation 
   
    # INPUT: range for imaginar and real axis 
    # OUTPUT: matrix containing complex numbers
    

    real_array = np.linspace(real_min, real_max, size, dtype=np.float32)
    imag_array = np.linspace(imag_min, imag_max, size, dtype=np.float32)
    
    complex_matrix = np.zeros((size, size), dtype=complex) # pre allocating 

    # Set up matrix with complex values
    for column in range(size):
        for row in range(size):
            complex_matrix[row, column] = complex(real_array[column], imag_array[row])

    return complex_matrix



def mandelbrot_computation(complex_nbr, threshold, iterations):
    
    # Takes in an array with complex numbers and does computation 
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
        if(i == iterations-1):
             mapped_entry = 1
   
    return mapped_entry



def map_array_dask(array, threshold, iterations, nbr_workers):
    
    """
    Function takes an array containing complex values and does computation 
    on each entry to check if its stable in terms of the mandelbrot set.
    
    Returns an array with linear mapped antries in the range [0, 1] 
    a value of 1 denotes a stable entry while a value below 1 is deemed ustable 
    """
    leng = len(array)
    client = Client(n_workers=nbr_workers) 
    start = time.time()
    result = client.map(mandelbrot_computation, array, [threshold]*leng, [iterations]*leng)
    # we want to minimize communicating results back to the local process. 
    # It’s often best to leave data on the cluster and operate on it remotely
        
    map_array = client.gather(result) # gather the results from the clients
    
    stop = time.time()
    client.close() # close the clients
    
    time_ex = stop - start
    return map_array, time_ex
    

def plot_mandelbrot(matrix, xmin, xmax, ymin, ymax):
        # we can PLOT the mandelbrot set

        fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})
    
        fig.suptitle("Mandelbrot set", fontweight='bold' )

        im = ax.imshow(matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation="bicubic")

        plt.colorbar(im, cax=cax, orientation='horizontal')
        plt.grid()
        plt.show()
        

def open_dask_status(url_string):
    # open the dask status 
    webbrowser.open_new(url_string)
    
                
if __name__ == '__main__':

    
    ITERATIONS = 200         # Number of iterations
    THRESHOLD = 2            # Threshol (radius on the unit circle)
    MATRIX_SIZE = 500        # Square matrix dimension
    WORKERS = 1             # Number of workers used for pool processing
    
    REAL_MIN = -2
    REAL_MAX = 1
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5
    
    start = time.time()
    complex_matrix = create_complex_matrix(REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
    stop = time.time()
    print('Generated matrix in:', stop-start, 'second(s)')
    print('Matrix size:', MATRIX_SIZE)
    
    # turn the generated matrix into 1D for mapping purposes
    complex_array = complex_matrix.flatten()
    
    map_array, time_execution = map_array_dask(complex_array, THRESHOLD, ITERATIONS, WORKERS)
    
    # turn the generated matrix  back into 2D for plotting purposes
    map_matrix =  np.reshape(map_array, (MATRIX_SIZE, MATRIX_SIZE)) 
    print('Mapped generated matrix in:', time_execution, 'second(s)')
    print('Using dask distributed with', WORKERS, 'workers')
    

    flag = input("Plot mandelbrot set? [y]/[n]")

    if flag == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX  )
    else:
        print("Done cu mate!")

    



    
    
   