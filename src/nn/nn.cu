#include "../linear-algebra/real.h"
#include "../linear-algebra/vec.h"
#include "../linear-algebra/matrix.h"
#include "../linear-algebra/cuda_matrix.h"
#include "../linear-algebra/cuda_vec.h"
#include "../linear-algebra/vec.cuh"
#include "../linear-algebra/matrix.cuh"
#include "../nn_layers/cuda_nn_layer.h"
#include "../nn_layers/activation_function.cuh"
#include "cuda_nn.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "../print/print.h"

void _swap_realptr_realptr(real **a, real **b) {
  real* tmp = *a;
  *a = *b;
  *b = tmp;
}

unsigned int _max_ui_ui_ui(unsigned int a, unsigned int b) {
  return (a > b) ? a : b;
}

__global__ void _cuda_nn_predict_step(
  const real* weights,
  const real* bias,
  const cuda_nn_layer layer,
  const real* input,
  const unsigned int input_size,
  real* output
) {
  cuda_matrix_apply(
    weights, input, output, layer.input_size, input_size
  );
  __syncthreads();
  cuda_vec_add(
    output, bias, output, layer.input_size
  );
  __syncthreads();
  // !!!WARN MAY BREAK IN THE FUTURE
  call_activation_function(output, layer.input_size, output, layer.output_size, layer.activation_function);
}

extern "C"
vec cuda_nn_predict(const cuda_nn nn_instance, const vec input) {
  int predict_step_min_grid_size, predict_step_block_size;
  cudaOccupancyMaxPotentialBlockSize(&predict_step_min_grid_size, &predict_step_block_size, _cuda_nn_predict_step);
  const unsigned int layers_count = nn_instance.layers_count;
  unsigned int max_size = input.size;
  for (unsigned int i = 0; i < layers_count; ++i) {
    const cuda_nn_layer layer = nn_instance.layers[i];
    max_size = _max_ui_ui_ui(
      max_size,
      _max_ui_ui_ui(layer.output_size, layer.input_size)
    );
  }
  real** weights = (real**) malloc(layers_count * sizeof(real*));
  real** bias = (real**) malloc(layers_count * sizeof(real*));
  for (unsigned int i = 0; i < layers_count; ++i) {
    weights[i] = matrix_to_cuda_matrix(nn_instance.weights[i]);
    bias[i] = vec_to_cuda_vec(nn_instance.bias[i]);
  }
  unsigned int current_signal_size = input.size;
  real* input_cuda_vec = vec_to_cuda_vec(input);
  real* current_signal;
  real* next_signal;
  cudaMalloc(&current_signal, max_size * sizeof(real));
  cudaMalloc(&next_signal, max_size * sizeof(real));
  cudaMemcpy(current_signal, input_cuda_vec, current_signal_size * sizeof(real), cudaMemcpyDeviceToDevice);
  cudaFree(input_cuda_vec);

  for (unsigned int i = 0; i < nn_instance.layers_count; ++i) {
    const cuda_nn_layer layer = nn_instance.layers[i];
    // TODO: Compute block_size and grid_size from input's size
    _cuda_nn_predict_step<<<1, 32>>>(
      weights[i], bias[i], layer, current_signal, current_signal_size, next_signal
    );
    cudaDeviceSynchronize();
    current_signal_size = layer.output_size;
    _swap_realptr_realptr(&current_signal, &next_signal);
  }

  vec result = vec_create(nn_instance.layers[layers_count - 1].output_size);
  cudaMemcpy(result.values, current_signal, result.size * sizeof(real), cudaMemcpyDeviceToHost);

  cudaFree(current_signal);
  cudaFree(next_signal);
  for (unsigned int i = 0; i < layers_count; ++i) {
    cudaFree(weights[i]);
    cudaFree(bias[i]);
  }
  free(weights);
  free(bias);

  return result;
}