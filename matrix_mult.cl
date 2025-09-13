__kernel void multiply(
    __global const double* A,    // Macierz wejściowa A (n x k)
    __global const double* B,    // Macierz wejściowa B (k x m)
    __global double* C,          // Macierz wyjściowa C (n x m)
    const int n,                 // Liczba wierszy A
    const int k,                 // Liczba kolumn A / wierszy B
    const int m)                 // Liczba kolumn B
{
    // Indeks kolumny w macierzy C
    int j = get_global_id(0);
    // Indeks wiersza w macierzy C
    int i = get_global_id(1);

    // Upewniamy się, że nie wychodzimy poza granice macierzy
    if (i < n && j < m) {
        double sum = 0.0;
        for (int l = 0; l < k; l++) {
            // C[i][j] += A[i][l] * B[l][j]
            sum += A[i * k + l] * B[l * m + j];
        }
        C[i * m + j] = sum;
    }
}