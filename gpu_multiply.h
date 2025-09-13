#ifndef GPU_MULTIPLY_H
#define GPU_MULTIPLY_H

#include "my_matrix.h"

/**
 * Mnoży macierze A i B na GPU przy użyciu OpenCL, wynik zapisuje w Result.
 * Wypisuje szczegółowe czasy operacji na GPU.
 */
int multiply_opencl(matrix_t *A, matrix_t *B, matrix_t *Result);

#endif // GPU_MULTIPLY_H