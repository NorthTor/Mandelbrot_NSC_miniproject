
# Tor Kaufmann Gjerde 

# dask_parallel_mandelbrot.py 
# A dask implementation for computing and plotting the mandelbrot set 


import time
import numpy as np
import matplotlib.pyplot as plt

from dask.distributed import Client, wait, LocalCluster 
                  

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



def mandelbrot_computation(complex_nbr, threshold, iterations):
    
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



def map_array_dask(array, threshold, iterations, nbr_workers):
    
    """
    Function takes an array containing complex values and does computation 
    on each entry to check if its stable in terms of the mandelbrot set.
    
    Returns an array with linear mapped antries in the range [0, 1] 
    a value of 1 denotes a stable entry while a value below 1 is deemed ustable 
    """
    
    size = len(array)
    
    cluster = LocalCluster( n_workers=nbr_workers,
                            processes=True, #default = True 
                            threads_per_worker=1)
    
    client = Client(cluster) 
    
    map_array = client.map(mandelbrot_computation, array, [threshold]*size, [iterations]*size)
    # we want to minimize communicating results back to the local process. 
    # It’s often best to leave data on the cluster and operate on it remotely
    wait(map_array) # wait for computation into map_array
    
    array = client.gather(map_array) # gather the results from the clients
    
    client.close() # close the clients
    
    return array
    

def plot_mandelbrot_set(matrix):
        # we can PLOT the mandelbrot set
        xmin, xmax = REAL_MATRIX_MIN, REAL_MATRIX_MAX
        ymin, ymax = IMAG_MATRIX_MIN, IMAG_MATRIX_MAX

        fig, (ax, cax) = plt.subplots(nrows=2, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [1, 0.05]})
    
        fig.suptitle("Mandelbrot set", fontweight='bold' )

        im = ax.imshow(matrix, cmap='hot', extent=[xmin, xmax, ymin, ymax],
                   interpolation="bicubic")

        plt.colorbar(im, cax=cax, orientation='horizontal')
        plt.grid()
        plt.show()
        
        
                
if __name__ == '__main__':

    
    ITERATIONS = 200         # Number of iterations
    THRESHOLD = 2            # Threshol (radius on the unit circle)
    MATRIX_SIZE = 200        # Square matrix dimension
    WORKERS = 2              # Number of workers used for pool processing
    
    REAL_MATRIX_MAX = 1
    REAL_MATRIX_MIN = -2
    IMAG_MATRIX_MIN = -1.5
    IMAG_MATRIX_MAX = 1.5
    
    start = time.time()
    complex_matrix = create_complex_matrix(REAL_MATRIX_MIN, REAL_MATRIX_MAX, IMAG_MATRIX_MIN,
                                   IMAG_MATRIX_MAX, MATRIX_SIZE)
    stop = time.time()
    print('Generated matrix in:', stop-start, 'second(s)')
    print('Matrix size:', MATRIX_SIZE)
    # Turn the generated matrix into 1D for mapping purposes
    complex_array = complex_matrix.flatten()
    
    start = time.time()
    map_array = map_array_dask(complex_array, THRESHOLD, ITERATIONS, WORKERS)
    stop = time.time()
    print(np.amin(map_array))
    
    print('Mapped generated matrix in:', stop-start, 'second(s)')
    print('Using', WORKERS, 'workers')
    # Turn the generated matrix  back into 2D for plotting purposes
    map_matrix =  np.reshape(map_array, (MATRIX_SIZE, MATRIX_SIZE)) 
    
    plot_mandelbrot_set(map_matrix)
    



    
    
   