#include "matmul.hpp"
#include <iostream>
#include <chrono>

int main() {
    const int N = 512;
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    fill_matrix(A, N, 1.0f);
    fill_matrix(B, N, 2.0f);

    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / (elapsed * 1e9);

    std::cout << "Time: " << elapsed << " s, GFLOPs: " << gflops << "\n";

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
