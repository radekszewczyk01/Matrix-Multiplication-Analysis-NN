// Plik: cuda_multiply.h

#ifndef CUDA_MULTIPLY_H
#define CUDA_MULTIPLY_H

#include "my_matrix.h"

// Ten blok mówi kompilatorowi C++, aby nie "dekorował" nazwy tej funkcji
#ifdef __cplusplus
extern "C" {
#endif

int multiply_cuda(matrix_t *A, matrix_t *B, matrix_t *Result);

#ifdef __cplusplus
}
#endif

#endif // CUDA_MULTIPLY_H