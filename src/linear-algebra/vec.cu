#include "real.h"
#include "vec.h"
#include <cuda_runtime.h>

__device__ void cuda_vec_add(const real* a, const real* b, real* c, unsigned int n) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__device__ void cuda_vec_subtract(const real* a, const real* b, real* c, unsigned int n) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] - b[idx];
  }
}

real* vec_to_cuda_vec(vec vector) {
  real* cuda_vector;
  size_t size = vector.size * sizeof(real);
  cudaMalloc(&cuda_vector, size);
  cudaMemcpy(cuda_vector, vector.values, size, cudaMemcpyHostToDevice);
  return cuda_vector;
}