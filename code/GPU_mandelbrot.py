""""
Author: Tor Kaufmann Gjerde
Generating and plotting of the Mandelbrot set
A GPU accelerated using PyOpenCl
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import h5py


def create_real_and_imag_matrices(real_min, real_max, imag_min, imag_max, size):
    """"
    Function that takes in a minimum and maximum value for the
    real and imaginary component of a complex number and returns
    two matrices one containing the real components and one containing the imaginary.
    The amount of steps between maximum and minimum values is linear with the size argument.
    The resulting real matrix obtains the same components in the vertical Y-direction
    across the whole matrix, and the resulting imaginary matrix obtains the same components
    in the horizontal X-direction across the whole matrix.

    :param real_min: Minimum value of real component
    :type real_min: float32
    :param real_max: Maximum value of real component
    :type real_max: float32
    :param imag_min: Minimum value of imaginary component
    :type imag_min: float32
    :param imag_max: Maximum value of imaginary component
    :type imag_max: float 32
    :param size: Size of the output matrix
    :type size: float32

    :return real_matrix: matrix with real components
    :return imag_matrix: Matrix with imaginary components
    :rtype real_matrix, imag_matrix: Numpy 2D array float32
    """

    real_array = np.linspace(real_min, real_max, size, dtype=np.float32)
    imag_array = np.linspace(imag_max, imag_min, size, dtype=np.float32)

    real_matrix = np.zeros((size, size), dtype=np.float32)  # pre allocating output vector
    imag_matrix = np.zeros((size, size), dtype=np.float32)  # pre allocating output vector

    # Set up matrix with complex values 
    for n in range(size):
        real_matrix[n, :] = real_array  # insert into output vector
        imag_matrix[:, n] = imag_array  # insert into output vector

    return real_matrix, imag_matrix


def mandelbrot_GPU(real_matrix, imag_matrix, iterations, threshold):
    """"
    Function that does GPU accelerated computation of the mandelbrot algorithm
    found in the separate kernel file

    :param real_matrix: A matrix containing all real components
    :type real_matrix: Numpy 2D array, float32
    :param imag_matrix: A matrix containing all imaginary components
    :type imag_matrix: Numpy 2D array, float32

    :return result_host: Matrix containing mapped values
    :rtype result_host: Numpy 2D array, float32
    """

    # Create the context (containing platform and device information)
    context = cl.create_some_context()
    # Kernel execution, synchronization, and memory transfer 
    # operations are submitted through the command que
    # each command queue points to a single device within a context.
    # Create command que:
    cmd_queue = cl.CommandQueue(context)

    real_matrix_host = real_matrix  # matrix containing real parts
    imag_matrix_host = imag_matrix  # matrix containing imaginary parts

    # Create empty matrix to hold the resulting mapped matrix
    result_host = np.empty((SIZE, SIZE)).astype(np.float32)

    # Create a device side read-only memory buffer and copy the data from "hostbuf" into it.
    mf = cl.mem_flags
    real_matrix_device = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=real_matrix_host)
    imag_matrix_device = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imag_matrix_host)

    result_device = cl.Buffer(context, mf.READ_WRITE, result_host.nbytes)

    # Source of the kernel itself.
    kernel_source = open("GPU_mandelbrot_kernel.cl").read()

    # Create a new program from the kernel and build the source.
    prog = cl.Program(context, kernel_source).build()

    # Execute the kernel in the program with parameters
    prog.mandelbrot(cmd_queue, (SIZE * SIZE,), None, real_matrix_device, imag_matrix_device,
                    result_device, np.int32(iterations), np.int32(threshold))

    # Copy the result back from device to host.
    cl.enqueue_copy(cmd_queue, result_host, result_device)

    return result_host


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

    fig.suptitle('Mandelbrot set - GPU accelerated', fontweight='bold')

    im = ax.imshow(matrix, cmap='hot', extent=[x_min, x_max, y_min, y_max],
                   interpolation='bicubic')

    plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.grid()
    plt.show()


if __name__ == '__main__':

    SIZE = 5000  # Square matrix size (order)
    ITERATIONS = 200  # Iterations for mandelbrot kernel
    THRESHOLD = 2  # Threshold used in mandelbrot kernel

    REAL_MIN = -2
    REAL_MAX = 1
    IMAG_MIN = -1.5
    IMAG_MAX = 1.5

    start1 = time.time()
    real_matrix, imag_matrix = create_real_and_imag_matrices(REAL_MIN, REAL_MAX,
                                                             IMAG_MIN, IMAG_MAX,
                                                             SIZE)
    stop1 = time.time()
    print('Generated Real and Imaginary matrix in:', stop1 - start1, 'second(s)')
    print('Square matrix size:', SIZE)

    start2 = time.time()
    map_matrix = mandelbrot_GPU(real_matrix, imag_matrix,
                                       ITERATIONS, THRESHOLD)
    stop2 = time.time()

    print('Mapped generated matrix in:', stop2 - start2, 'second(s)')

    flag_save = input("Save output (mapped matrix)? [y]/[n]")
    if flag_save == "y":
        file = h5py.File("GPU_mandelbrot.hdf5", "w")
        file.create_dataset('dataset', data=map_matrix)
        file.close()

    flag_plot = input("Plot mandelbrot set? [y]/[n]")
    if flag_plot == "y":
        print("Loading.. please wait")
        plot_mandelbrot(map_matrix, REAL_MIN, REAL_MAX, IMAG_MIN, IMAG_MAX)

    print("Done cu mate!")
