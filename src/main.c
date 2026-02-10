#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn/nn.h"
#include "nn_layers/sigmoid_layer.h"
#include "nn_layers/relu_layer.h"
#include "nn_layers/identity_layer.h"
#include "print/print.h"
#include "nn/cuda_nn.h"

int main(void) {

  nn_layer _tmp_layers[] = {
    create_relu_layer(10),
    create_identity_layer(10)
  };
  nn_layer *layers = malloc(sizeof(_tmp_layers));
  memcpy(layers, _tmp_layers, sizeof(_tmp_layers));
  nn nn_instance = nn_rand(layers, sizeof(_tmp_layers)/sizeof(nn_layer), 10, 10);
  vec input = vec_create(10);
  vec output = vec_create(10);
  for (unsigned int i = 0; i < 10; ++i) {
    const float x = (float)i;
    input.values[i] = x;
    output.values[i] = 2 * x - 3;
  }
  nn_fit(nn_instance, input, output, 5000);
  vec_free(input);
  vec_free(output);
  vec test_input = vec_create(10);
  for (unsigned int i = 0; i < 10; ++i) {
    test_input.values[i] = (float)i;
  }
  vec result = nn_predict(nn_instance, test_input);
  print_vec(result);
  printf("\n");

  cuda_nn_layer cuda_layers[] = {
    {
      .input_size = 10,
      .output_size = 10,
      .activation_function = activation_RELU
    },
    {
      .input_size = 10,
      .output_size = 10,
      .activation_function = activation_identity
    }
  };
  vec result2 = cuda_nn_predict((cuda_nn) {
    .bias = nn_instance.bias,
    .weights = nn_instance.weights,
    .layers_count = nn_instance.layers_count,
    .output_size = nn_instance.output_size,
    .input_size = nn_instance.input_size,
    .layers = cuda_layers
  }, test_input);
  print_vec(result2);
  printf("\n");

  vec_free(result2);
  vec_free(test_input);
  vec_free(result);
  nn_free(nn_instance);

  return 0;
}