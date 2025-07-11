#ifndef VEC_CUH
#define VEC_CUH

#include "real.h"

__device__ void cuda_vec_add(const real* a, const real* b, real* c, unsigned int n);

__device__ void cuda_vec_subtract(const real* a, const real* b, real* c, unsigned int n);

#endif