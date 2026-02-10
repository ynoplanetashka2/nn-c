#ifndef ACTIVATION_FUNCTION_CUH
#define ACTIVATION_FUNCTION_CUH

#include "cuda_activation_function.h"
#include "../linear-algebra/real.h"

__global__ void call_activation_function(
  const real* argument,
  const unsigned int argument_size,
  real* output,
  const unsigned int output_size,
  const activation_fn_type fn_type
);

__device__ void call_activation_function_derivative(
  const real* argument,
  const unsigned int argument_size,
  real* output,
  const unsigned int output_size,
  const activation_fn_type fn_type
);

#endif