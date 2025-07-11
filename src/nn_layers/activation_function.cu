#include "cuda_activation_function.h"
#include <math.h>
#include "../linear-algebra/real.h"


__device__ void _activation_identity(
  const real* argument,
  const unsigned int argument_size,
  real* output,
  const unsigned int output_size
) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < output_size) {
    output[i] = argument[i];
  }
}

__device__ void _activation_RELU(
  const real* argument,
  const unsigned int argument_size,
  real* output,
  const unsigned int output_size
) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < output_size) {
    const real x = argument[i];
    output[i] = x > 0 ? x : 0;
  }
}

__device__ void _activation_sigmoid(
  const real* argument,
  const unsigned int argument_size,
  real* output,
  const unsigned int output_size
) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < output_size) {
    const real x = argument[i];
    output[i] = 1.0 / (1 + exp(-x));
  }
}

__device__ void call_activation_function(
  const real* argument,
  const unsigned int argument_size,
  real* output,
  const unsigned int output_size,
  const activation_fn_type fn_type
) {
  switch (fn_type) {
    case activation_identity:
      _activation_identity(argument, argument_size, output, output_size);
      break;
    case activation_RELU:
      _activation_RELU(argument, argument_size, output, output_size);
      break;
    case activation_sigmoid:
      _activation_sigmoid(argument, argument_size, output, output_size);
      break;
  }
}

__device__ void call_activation_function_derivative(
  const real* argument,
  const unsigned int argument_size,
  real* output,
  const unsigned int output_size,
  const activation_fn_type fn_type
) {
  return;
}