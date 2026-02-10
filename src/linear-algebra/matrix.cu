#include "real.h"
#include "matrix.h"
#include <cuda_runtime.h>

__device__ void cuda_matrix_multiply(
  const real* a, const real* b, real* output,
  const unsigned int a_height, const unsigned int a_width,
  const unsigned int b_height, const unsigned int b_width
) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < a_height && col < b_width) {
    real sum = 0.0;
    for (unsigned int k = 0; k < a_width; ++k) {
      sum += a[row * a_width + k] * b[k * b_width + col];
    }
    output[row * b_width + col] = sum;
  }
}

__global__ void cuda_matrix_apply(
  const real* matrix, const real* vector, real* output,
  const unsigned int height, const unsigned int width
) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < height) {
    real sum = 0.0;
    for (unsigned int i = 0; i < width; ++i) {
      sum += vector[i] * matrix[idx * width + i];
    }
    output[idx] = sum;
  }
}

real* matrix_to_cuda_matrix(const matrix mat) {
  real* cuda_matrix;
  size_t size = mat.height * mat.width * sizeof(real);
  cudaMalloc(&cuda_matrix, size);
  for (unsigned int i = 0; i < mat.height; ++i) {
    cudaMemcpy(&cuda_matrix[i * mat.width], mat.values[i], sizeof(real) * mat.width, cudaMemcpyHostToDevice);
  }
  return cuda_matrix;
}