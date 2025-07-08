#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "../linear-algebra/vec.h"
#include "../linear-algebra/matrix.h"

typedef vec (*transform_fn)(unsigned int input_size, unsigned int output_size, vec vector);
typedef matrix (*transform_fn_derivative)(unsigned int input_size, unsigned int output_size, vec vector);

typedef struct {
  unsigned int input_size;
  unsigned int output_size;
  transform_fn transform;
  transform_fn_derivative transform_derivative;
} nn_layer;

#endif