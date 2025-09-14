#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpu_multiply.h"

#define KERNEL_FILE "matrix_mult.cl"

// Prosta makro do sprawdzania błędów
#define CHECK_CL_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "Błąd OpenCL (%d) w '%s'\n", err, msg); \
        return -1; \
    }

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
    cl_int err; // Zmienna do przechowywania kodów błędów

    // 1. Konfiguracja OpenCL
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL_ERROR(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_CL_ERROR(err, "clGetDeviceIDs - upewnij się, że masz GPU z obsługą OpenCL!");

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_CL_ERROR(err, "clCreateCommandQueue");

    const int n = A->n_rows;
    const int k = A->n_cols;
    const int m = B->n_cols;

    // 2. Tworzenie buforów na GPU
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * k, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateBuffer d_A");
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * k * m, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateBuffer d_B");
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * m, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateBuffer d_C");

    // 3. Kopiowanie danych na GPU
    cl_event event_write_A, event_write_B, event_kernel, event_read_C;
    err = clEnqueueWriteBuffer(queue, d_A, CL_FALSE, 0, sizeof(float) * n * k, A->matrix, 0, NULL, &event_write_A);
    CHECK_CL_ERROR(err, "clEnqueueWriteBuffer d_A");
    err = clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, sizeof(float) * k * m, B->matrix, 0, NULL, &event_write_B);
    CHECK_CL_ERROR(err, "clEnqueueWriteBuffer d_B");

    // 4. Kompilacja kernela
    char* kernel_source = load_kernel_source(KERNEL_FILE);
    if (!kernel_source) {
        fprintf(stderr, "Błąd: Nie udało się wczytać pliku kernela %s\n", KERNEL_FILE);
        return -1;
    }
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &err);
    CHECK_CL_ERROR(err, "clCreateProgramWithSource");
    free(kernel_source);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Błąd kompilacji kernela OpenCL:\n%s\n", build_log);
        return -1;
    }

    // 5. Ustawienie argumentów i uruchomienie kernela
    cl_kernel kernel = clCreateKernel(program, "multiply", &err);
    CHECK_CL_ERROR(err, "clCreateKernel");
    
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A); CHECK_CL_ERROR(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B); CHECK_CL_ERROR(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C); CHECK_CL_ERROR(err, "clSetKernelArg 2");
    err = clSetKernelArg(kernel, 3, sizeof(int), &n);     CHECK_CL_ERROR(err, "clSetKernelArg 3");
    err = clSetKernelArg(kernel, 4, sizeof(int), &k);     CHECK_CL_ERROR(err, "clSetKernelArg 4");
    err = clSetKernelArg(kernel, 5, sizeof(int), &m);     CHECK_CL_ERROR(err, "clSetKernelArg 5");

    size_t global_work_size[2] = {m, n};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event_kernel);
    CHECK_CL_ERROR(err, "clEnqueueNDRangeKernel");

    // 6. Kopiowanie wyników z GPU
    err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * n * m, Result->matrix, 0, NULL, &event_read_C);
    CHECK_CL_ERROR(err, "clEnqueueReadBuffer d_C");

    clFinish(queue);

    // 7. Pomiar i wypisanie czasów (te funkcje też zwracają błędy!)
    cl_ulong start, end;
    clGetEventProfilingInfo(event_write_A, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event_write_B, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    float transfer_to_gpu_ms = (end - start) / 1e6;

    clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event_kernel, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    float kernel_ms = (end - start) / 1e6;

    clGetEventProfilingInfo(event_read_C, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event_read_C, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    float transfer_from_gpu_ms = (end - start) / 1e6;

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