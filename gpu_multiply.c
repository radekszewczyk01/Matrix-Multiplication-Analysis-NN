#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_multiply.h"

#define KERNEL_FILE "matrix_mult.cl"

// Funkcja pomocnicza do wczytania kernela z pliku
static char* load_kernel_source(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) { return NULL; }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    char* source = (char*)malloc(size + 1);
    (void)fread(source, 1, size, file);
    source[size] = '\0';
    fclose(file);
    return source;
}

int multiply_opencl(matrix_t *A, matrix_t *B, matrix_t *Result) {
    // 1. Konfiguracja OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Pobranie wymiarów
    const int n = A->n_rows;
    const int k = A->n_cols;
    const int m = B->n_cols;

    // 2. Tworzenie buforów na GPU
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * n * k, NULL, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * k * m, NULL, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * n * m, NULL, NULL);

    // 3. Kopiowanie danych na GPU (z pomiarem czasu)
    cl_event event_write_A, event_write_B, event_kernel, event_read_C;
    clEnqueueWriteBuffer(queue, d_A, CL_FALSE, 0, sizeof(double) * n * k, A->matrix, 0, NULL, &event_write_A);
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, sizeof(double) * k * m, B->matrix, 0, NULL, &event_write_B);

    // 4. Kompilacja kernela
    char* kernel_source = load_kernel_source(KERNEL_FILE);
    if (!kernel_source) {
        fprintf(stderr, "Error: Failed to load kernel source from %s\n", KERNEL_FILE);
        return -1;
    }
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, NULL);
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
        char build_log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Błąd kompilacji kernela:\n%s\n", build_log);
        free(kernel_source);
        return -1;
    }
    free(kernel_source);

    // 5. Ustawienie argumentów i uruchomienie kernela
    cl_kernel kernel = clCreateKernel(program, "multiply", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &n);
    clSetKernelArg(kernel, 4, sizeof(int), &k);
    clSetKernelArg(kernel, 5, sizeof(int), &m);

    size_t global_work_size[2] = {m, n}; // {liczba kolumn C, liczba wierszy C}
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event_kernel);

    // 6. Kopiowanie wyników z GPU
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(double) * n * m, Result->matrix, 0, NULL, &event_read_C);
    clFinish(queue);

    // 7. Pomiar i wypisanie czasów
    cl_ulong start, end;
    clGetEventProfilingInfo(event_write_A, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event_write_B, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double transfer_to_gpu_ms = (end - start) / 1e6;

    clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double kernel_ms = (end - start) / 1e6;

    clGetEventProfilingInfo(event_read_C, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event_read_C, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double transfer_from_gpu_ms = (end - start) / 1e6;

    printf("    -> Czas kopiowania na GPU: %.3f ms\n", transfer_to_gpu_ms);
    printf("    -> Czas wykonania kernela: %.3f ms\n", kernel_ms);
    printf("    -> Czas kopiowania z GPU:  %.3f ms\n", transfer_from_gpu_ms);

    // 8. Zwolnienie zasobów
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}