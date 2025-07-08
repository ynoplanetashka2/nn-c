#ifndef OPERATION_H
#define OPERATION_H

#include "vec.h"

typedef vec (*transform_fn)(vec input);

typedef struct {
  transform_fn operation;
  transform_fn derivative;
} differentiable_operation;

#endif