#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "real.h"

__device__ void cuda_matrix_multiply(
  const real* a, const real* b, real* output,
  const unsigned int a_height, const unsigned int a_width,
  const unsigned int b_height, const unsigned int b_width
);

__global__ void cuda_matrix_apply(
  const real* matrix, const real* vector, real* output,
  const unsigned int height, const unsigned int width
);

#endif