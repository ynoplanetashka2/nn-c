#ifndef CUDA_NN_H
#define CUDA_NN_H

#include "../linear-algebra/matrix.h"
#include "../linear-algebra/vec.h"
#include "../nn_layers/cuda_nn_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  /**
   * layers count excluding input one. must be greater than 0
   */
  unsigned int layers_count;
  unsigned int input_size;
  unsigned int output_size;
  cuda_nn_layer* layers;
  matrix* weights;
  vec* bias;
} cuda_nn;

vec cuda_nn_predict(const cuda_nn nn_instance, const vec input);

#ifdef __cplusplus
}
#endif

#endif