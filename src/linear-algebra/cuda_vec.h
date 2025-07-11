#ifndef CUDA_VEC_H
#define CUDA_VEC_H

#include "real.h"

/**
 * Converts a vector from host memory to device memory.
 */
real* vec_to_cuda_vec(vec vector);

#endif