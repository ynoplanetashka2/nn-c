#include "identity_layer.h"
#include "../linear-algebra/real.h"

real identity(real arg) {
  return arg;
}

real identity_derivative(real arg) {
  return 1;
}

vec identity_transform(unsigned int input_size, unsigned int output_size, vec vector) {
  vec result = vec_create(output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i] = identity(vector.values[i]);
  }
  return result;
}

matrix identity_transform_derivative(unsigned int input_size, unsigned int output_size, vec vector) {
  matrix result = matrix_zeros(output_size, output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i][i] = identity_derivative(vector.values[i]);
  }
  return result;
}

nn_layer create_identity_layer(unsigned int size) {
  return (nn_layer) {
    .input_size = size,
    .output_size = size,
    .transform = identity_transform,
    .transform_derivative = identity_transform_derivative
  };
}

