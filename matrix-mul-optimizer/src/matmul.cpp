#include "matmul.hpp"
#include <cstring>

//Function to brute-force multiply matrices A and B, C is the output matrix and N is the size(NxN)
//Uses a nested loop O(n2)
void matmul_naive(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, sizeof(float) * N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}


//Fills matrix M of size NxN with value
void fill_matrix(float* M, int N, float value) {
    for (int i = 0; i < N * N; ++i)
        M[i] = value;
}


//Checks if two matrices are equal
bool matrices_equal(const float* A, const float* B, int N, float epsilon) {
    for (int i = 0; i < N * N; ++i)
        if (std::abs(A[i] - B[i]) > epsilon)
            return false;
    return true;
}
