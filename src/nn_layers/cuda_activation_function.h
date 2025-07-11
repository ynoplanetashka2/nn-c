#ifndef CUDA_ACTIVATION_FUNCTION_H
#define CUDA_ACTIVATION_FUNCTION_H

typedef enum {
  activation_sigmoid,
  activation_RELU,
  activation_identity,
  activation_softmax
} activation_fn_type;

#endif