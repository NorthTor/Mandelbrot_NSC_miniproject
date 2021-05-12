// Author: Tor K. Gjerde
// Mandelbrot kernel for the GPU_mandelbrot.py script 

// #define PYOPENCL_DEFINE_CDOUBLE  // Uncomment if using double precision on supported device
                                    // and change "cfloat_t" to "cdouble_t" in following code 
                                    // NOT supported on Macbook Pro A2015
#include <pyopencl-complex.h>       


  __kernel void mandelbrot(
    __global const float *real_matrix_device, 
    __global const float *imag_matrix_device, 
    __global       float *result_device,
             ushort const max_iter,
             ushort const threshold)
    
{
  // Do computation with complex number
  // We import the real and imaginary components for the 
  // complex number from the two imput matricess
     
  int id = get_global_id(0);
  
  cfloat_t Z;  // value  
  cfloat_t C;  //value
  float abs;
  
  // Do mandelbrot computation and mapping
  C = cfloat_new(real_matrix_device[id], imag_matrix_device[id]); // get the complex number
  Z = cfloat_new(0, 0); // initialize complex number Z to be zero
  
  // Do iterations on the complex number
  bool flag = true;
  for(int i = 0; i < max_iter; i++){
     
      Z = cfloat_mul(Z,Z); // same as raise to power of two
      Z = cfloat_add(Z,C); // add the complex value C as in the mandelbrot
      
      abs = cfloat_abs(Z);
      // check to see if the iteration has "exploded"
      if(abs > float(threshold)){ // yes it exploded --> do mapping 
          flag = false; 
          result_device[id] = abs / float(max_iter); 
          break; 
      }
      if(i == max_iter - 1){
          // the iteration did not "explode" - therefore marked as stable wit a "1"
          result_device[id] = 1;
      } 
  }
}

