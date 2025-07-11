#ifndef PRINT_H
#define PRINT_H

#include "../linear-algebra/vec.h"
#include "../linear-algebra/matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

void print_vec(const vec vector);

void print_matrix(const matrix mat);

#ifdef __cplusplus
}
#endif

#endif