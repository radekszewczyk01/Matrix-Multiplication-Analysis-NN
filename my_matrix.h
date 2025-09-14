#ifndef MY_MATRIX_H // Jeśli MY_MATRIX_H nie jest zdefiniowane...
#define MY_MATRIX_H

typedef struct {
    float *matrix;
    int n_rows;
    int n_cols;
} matrix_t;


matrix_t *create_random_matrix(int rows, int cols);


/**
 * After last matrix it is required to provide NULL argument
 */
void destroy_matrices(matrix_t *first_matrix, ...);


void clean_matrix(matrix_t *A);

#endif
