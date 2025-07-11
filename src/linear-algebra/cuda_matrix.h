#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H

#include "real.h"
#include "matrix.h"

/**
 * Converts a matrix from host memory to device memory.
 */
real* matrix_to_cuda_matrix(const matrix mat);

#endif