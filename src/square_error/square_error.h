#ifndef SQUARE_ERROR_H
#define SQUARE_ERROR_H

#include "../linear-algebra/vec.h"

float square_error_vec(vec vec1, vec vec2);

vec square_error_vec_gradient(vec actual_value, vec expected_value);

#endif