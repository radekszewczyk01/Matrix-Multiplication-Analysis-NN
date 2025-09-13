#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>

#include "my_matrix.h"

matrix_t *create_random_matrix(int rows, int cols)
{
    matrix_t *res = (matrix_t *)malloc(sizeof(matrix_t));
    if (res == NULL) {
        fprintf(stderr, "Error memory allocation for matrix_t struct\n");
        return NULL;
    }

    double *matrix = (double *)malloc(rows * cols * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Error memory allocation\n");
        return NULL;
    }

    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }

    res->matrix = matrix;
    res->n_rows = rows;
    res->n_cols = cols;

    return res;
}


/**
 * After last matrix it is required to provide NULL argument
 */
void destroy_matrices(matrix_t *first_matrix, ...)
{
    if (first_matrix == NULL) {
        return;
    }

    free(first_matrix->matrix);
    free(first_matrix);

    va_list args;
    va_start(args, first_matrix);

    matrix_t *current_matrix;
    while ((current_matrix = va_arg(args, matrix_t *)) != NULL) {
        free(current_matrix->matrix);
        free(current_matrix);
    }

    va_end(args);
}


void clean_matrix(matrix_t *A)
{
    assert(A && A->matrix);

    long long total_elements = (long long)A->n_rows * A->n_cols;
    for(int i = 0; i < total_elements; ++i) {
        A->matrix[i] = 0;
    }
}
