#include "relu_layer.h"
#include "../linear-algebra/real.h"

real RELU(real arg) {
  if (arg < 0) {
    return 0;
  }
  return arg;
}

real RELU_derivative(real arg) {
  if (arg < 0) {
    return 0;
  }
  return 1;
}

vec RELU_transform(unsigned int input_size, unsigned int output_size, vec vector) {
  vec result = vec_create(output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i] = RELU(vector.values[i]);
  }
  return result;
}

matrix RELU_transform_derivative(unsigned int input_size, unsigned int output_size, vec vector) {
  matrix result = matrix_zeros(output_size, output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i][i] = RELU_derivative(vector.values[i]);
  }
  return result;
}

nn_layer create_relu_layer(unsigned int size) {
  return (nn_layer) {
    .input_size = size,
    .output_size = size,
    .transform = RELU_transform,
    .transform_derivative = RELU_transform_derivative
  };
}
