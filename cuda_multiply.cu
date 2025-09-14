#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_multiply.h"

// Funkcja pomocnicza do sprawdzania błędów CUDA (bez zmian)
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Błąd CUDA: %s - %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

extern "C" int multiply_cuda(matrix_t *A, matrix_t *B, matrix_t *Result) {
    // --- NOWOŚĆ: Tworzenie obiektów zdarzeń ---
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");
    
    // 1. Inicjalizacja
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int n = A->n_rows;
    const int k = A->n_cols;
    const int m = B->n_cols;

    float *d_A, *d_B, *d_C;

    // 2. Alokacja pamięci na GPU
    checkCudaError(cudaMalloc((void**)&d_A, sizeof(float) * n * k), "Alokacja d_A");
    checkCudaError(cudaMalloc((void**)&d_B, sizeof(float) * k * m), "Alokacja d_B");
    checkCudaError(cudaMalloc((void**)&d_C, sizeof(float) * n * m), "Alokacja d_C");

    // --- NOWOŚĆ: Pomiar czasu transferu na GPU ---
    float transfer_to_gpu_ms, kernel_ms, transfer_from_gpu_ms;
    
    cudaEventRecord(start); // Umieść znacznik 'start' w kolejce
    checkCudaError(cudaMemcpy(d_A, A->matrix, sizeof(float) * n * k, cudaMemcpyHostToDevice), "Kopiowanie A na GPU");
    checkCudaError(cudaMemcpy(d_B, B->matrix, sizeof(float) * k * m, cudaMemcpyHostToDevice), "Kopiowanie B na GPU");
    cudaEventRecord(stop); // Umieść znacznik 'stop' w kolejce
    cudaEventSynchronize(stop); // Zaczekaj, aż GPU dojdzie do znacznika 'stop'
    cudaEventElapsedTime(&transfer_to_gpu_ms, start, stop); // Oblicz czas, jaki minął między nimi

    // 4. Wywołanie mnożenia macierzy z pomiarem czasu
    const float alpha = 1.0;
    const float beta = 0.0;
    
    cudaEventRecord(start); // Ponownie użyj znacznika 'start'
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_B, m, d_A, k, &beta, d_C, m);
    cudaEventRecord(stop); // I 'stop'
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // 5. Kopiowanie wyniku z GPU na CPU z pomiarem czasu
    cudaEventRecord(start);
    checkCudaError(cudaMemcpy(Result->matrix, d_C, sizeof(float) * n * m, cudaMemcpyDeviceToHost), "Kopiowanie C z GPU");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_from_gpu_ms, start, stop);

    // --- NOWOŚĆ: Wypisanie szczegółowych czasów ---
    printf("    -> Czas kopiowania na GPU: %.3f ms\n", transfer_to_gpu_ms);
    printf("    -> Czas wykonania kernela: %.3f ms\n", kernel_ms);
    printf("    -> Czas kopiowania z GPU:  %.3f ms\n", transfer_from_gpu_ms);

    // 6. Zwolnienie zasobów
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}