#ifndef NN_H
#define NN_H

#include "../linear-algebra/vec.h"
#include "../linear-algebra/matrix.h"
#include "../nn_layers/nn_layer.h"

typedef struct {
  /**
   * layers count excluding input one. must be greater than 0
   */
  unsigned int layers_count;
  unsigned int input_size;
  unsigned int output_size;
  nn_layer* layers;
  matrix* weights;
  vec* bias;
} nn;

nn nn_create(nn_layer* layers, unsigned int layers_count, unsigned int input_size, unsigned int output_size);

void nn_free(nn nn_instance);

nn nn_zeros(nn_layer* layers, unsigned int layers_count, unsigned int input_size, unsigned int output_size);

nn nn_rand(nn_layer* layers, unsigned int layers_count, unsigned int input_size, unsigned int output_size);

void nn_fit(nn nn_instance, const vec input, const vec expected_output, unsigned int training_cycles);

vec nn_predict(const nn nn_instance, const vec input);

#endif