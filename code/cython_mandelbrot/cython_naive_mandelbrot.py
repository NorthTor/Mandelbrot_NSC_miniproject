"""
@author: Tor Kaufmann Gjerde  
May 201

Followig code run the compiled code from cython_naive_mandelbrot.pyx

Other dependecies:
    cython_naive_mandelbrot.c
    
"""
import cython_naive_mandelbrot
import time


ITERATIONS = 200    # Number of iterations
THRESHOLD = 2       # Threshol
MATRIX_SIZE = 5000   # Square matrix dimension

REAL_MAX = 1
REAL_MIN = -2
IMAG_MIN = -1.5
IMAG_MAX = 1.5

start = time.time()
real = cython_naive_mandelbrot.create_matrix_real(REAL_MIN, REAL_MAX, MATRIX_SIZE)

imag = cython_naive_mandelbrot.create_matrix_imaginary(IMAG_MIN, IMAG_MAX, MATRIX_SIZE)
stop  = time.time()

print('Generated RE and IM matrices in:', stop-start, 'second(s)')
print('Square matrix size:', MATRIX_SIZE)

print("Mapping mandelbrot please wait...")

start = time.time()
map_matrix = cython_naive_mandelbrot.map_matrix(real, imag, ITERATIONS, THRESHOLD)
stop = time.time()

print('Mapped mandelbrot set in:', stop-start, 'second(s)')

flag = input("Plot mandelbrot set? [y]/[n]")

if flag == "y":
    print("Loading.. please wait")
    cython_naive_mandelbrot.plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX  )
else:
    print("Done cu to")
     