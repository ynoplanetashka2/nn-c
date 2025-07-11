#ifndef CUDA_NN_LAYER_H
#define CUDA_NN_LAYER_H

#include "cuda_activation_function.h"

typedef struct {
  activation_fn_type activation_function;
  unsigned int input_size;
  unsigned int output_size;
} cuda_nn_layer;

#endif