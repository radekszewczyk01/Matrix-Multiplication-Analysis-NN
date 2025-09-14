#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>
#include <assert.h>
#include <omp.h>
#include <cblas.h>

#include "my_matrix.h"
#include "gpu_multiply.h" 
#include "cuda_multiply.h" 

int dimensions_check_for_mul(matrix_t *A, matrix_t *B, matrix_t *Result){
    if (A->n_cols != B->n_rows) {
        return -1;
    }

    if (A->n_rows != Result->n_rows) {
        return -1;
    }

    if (B->n_cols != Result->n_cols) {
        return -1;
    }

    return 0;
}


int check_matrices_for_mul(matrix_t *A, matrix_t *B, matrix_t *Result, char *funcName)
{
    assert(A && B && Result);

    if (dimensions_check_for_mul(A, B, Result) < 0) {
        fprintf(stderr, "Error: %s - incompatible dimensions of matrices\n", funcName);
        return -1;
    }
    return 0;
}


/**
 * Inefective use of cashe memory
 */
int multiply_naive(matrix_t *A, matrix_t *B, matrix_t *Result)
{
    if (check_matrices_for_mul(A, B, Result, "multiply_naive()") < 0) {
        return - 1;
    }

    for(int i = 0; i < A->n_rows; ++i) {
        for (int j = 0; j < B->n_cols; j++) {
            for (int k = 0; k < A->n_cols; ++k) {
                Result->matrix[i * B->n_cols + j] += A->matrix[i * A->n_cols + k] * B->matrix[k * B->n_cols + j]; 
            }
        }
    }
    return 0;
}


/**
 * Optimal usage of cashe
 */
int multiply_ikj(matrix_t *A, matrix_t *B, matrix_t *Result)
{
    if (check_matrices_for_mul(A, B, Result, "multiply_ikj()") < 0) {
        return - 1;
    }

    for (int i  = 0; i < A->n_rows; ++i) {
        for (int l = 0; l < B->n_rows; ++l) {
            float r = A->matrix[i * A->n_cols + l];
            for (int j = 0; j < B->n_cols; ++j) {
                Result->matrix[i * B->n_cols + j] += r * B->matrix[l * B->n_cols + j];
            }
        }
    }
    return 0;
}


int multiply_blocked(matrix_t *A, matrix_t *B, matrix_t *Result)
{
    const int BLOCK_SIZE = 32;

    if (check_matrices_for_mul(A, B, Result, "multiply_blocked()") < 0) {
        return - 1;
    }

    // Big blocks
    for (int i0 = 0; i0 < A->n_rows; i0 += BLOCK_SIZE) {
        for (int l0 = 0; l0 < A->n_cols; l0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 <= B->n_cols; j0 += BLOCK_SIZE) {
                //Small blocks
                for (int i = i0; i < i0 + BLOCK_SIZE && i < A->n_rows; ++i) {
                    for (int l = l0; l < l0 + BLOCK_SIZE && l < A->n_cols; ++l) {
                        float r = A->matrix[i * A->n_cols + l];
                        for (int j = j0; j < j0 + BLOCK_SIZE && j < B->n_cols; ++j) {
                            Result->matrix[i*B->n_cols + j] += r * B->matrix[l * B->n_cols + j];
                        }
                    }
                }
            }
        }
    }

    return 0;
}


int multiply_parallel(matrix_t *A, matrix_t *B, matrix_t *Result)
{
    if (check_matrices_for_mul(A, B, Result, "multiply_parallel()") < 0) {
        return - 1;
    }

    #pragma omp parallel for
    for (int i  = 0; i < A->n_rows; ++i) {
        for (int l = 0; l < B->n_rows; ++l) {
            float r = A->matrix[i * A->n_cols + l];
            for (int j = 0; j < B->n_cols; ++j) {
                Result->matrix[i * B->n_cols + j] += r * B->matrix[l * B->n_cols + j];
            }
        }
    }
    return 0;
}


int multiply_blas(matrix_t *A, matrix_t *B, matrix_t *Result)
{
    if (check_matrices_for_mul(A, B, Result, "multiply_blas()") < 0) {
        return - 1;
    }

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A->n_rows, B->n_cols, A->n_cols,
        1.0,
        A->matrix, A->n_cols,
        B->matrix, B->n_cols,
        0.0,
        Result->matrix, Result->n_cols
    );
    return 0;
}


void calc_and_print_time(char *funcName, struct timespec *start, struct timespec *end)
{
    float elapsed_time = (end->tv_sec - start->tv_sec) * 1e3 + (end->tv_nsec - start->tv_nsec) * 1e-6;
    printf("Function: %s\nTime measured: %f miliseconds\n", funcName, elapsed_time);
}


typedef int (*multiply_func_t)(matrix_t*, matrix_t*, matrix_t*);

int benchmark_multiply(multiply_func_t func_to_run, char *func_name, int n, int k, int m)
{
    struct timespec start, end;
    int status;

    printf("--- Running test for: %s ---\n", func_name);
    matrix_t *A = create_random_matrix(n, k);
    matrix_t *B = create_random_matrix(k, m);
    matrix_t *C = create_random_matrix(n, m);

    if (!A || !B || !C) {
        destroy_matrices(A, B, C, NULL);
        return -1;
    }

    clean_matrix(C);
    clock_gettime(CLOCK_MONOTONIC, &start);

    status = func_to_run(A, B, C);

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (status < 0) {
        fprintf(stderr, "Error in function: %s\n", func_name);
        return -1;
    }

    calc_and_print_time(func_name, &start, &end);

    destroy_matrices(A, B, C, NULL);
    printf("----------------------------------------\n\n");
    return 0;
}


int main()
{
    int factor = 3;
    int n = 4096 * factor;
    int k = 4096 * factor;
    int m = 4096 * factor;

    // benchmark_multiply(multiply_naive, "multiply_naive()", n, k, m);
    // benchmark_multiply(multiply_ikj, "multiply_ikj()", n, k, m);
    // benchmark_multiply(multiply_blocked, "multiply_blocked()", n, k, m);
    // benchmark_multiply(multiply_parallel, "multiply_parallel()", n, k, m);
    benchmark_multiply(multiply_blas, "multiply_blas()", n, k, m);
    // benchmark_multiply(multiply_opencl, "multiply_opencl() (GPU OpenCL)", n, k, m);
    benchmark_multiply(multiply_cuda, "multiply_cuda() (GPU CUDA/cuBLAS)", n, k, m);
    
    return 0;
}