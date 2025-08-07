// Optimized CPU GEMM using cache blocking + OpenMP threading.
// Usage: ./gemm_cpu [m n k]  (defaults 1024 1024 1024)

#include <bits/stdc++.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

void gemm_blocked_omp(const double* A, const double* B, double* C,
                      int M, int N, int K, int BS=128)
{
  // Assume row-major storage:
  // A[M x K], B[K x N], C[M x N]
  // C += A * B
  #pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 0; ii < M; ii += BS) {
    for (int jj = 0; jj < N; jj += BS) {
      for (int kk = 0; kk < K; kk += BS) {
        int iimax = std::min(ii + BS, M);
        int jjmax = std::min(jj + BS, N);
        int kkmax = std::min(kk + BS, K);
        for (int i = ii; i < iimax; ++i) {
          for (int j = jj; j < jjmax; ++j) {
            double s = C[i*N + j];
            // Unroll the innermost loop a bit for better auto-vectorization
            int p = kk;
            for (; p + 4 <= kkmax; p += 4) {
              s += A[i*K + p + 0] * B[(p + 0)*N + j];
              s += A[i*K + p + 1] * B[(p + 1)*N + j];
              s += A[i*K + p + 2] * B[(p + 2)*N + j];
              s += A[i*K + p + 3] * B[(p + 3)*N + j];
            }
            for (; p < kkmax; ++p) {
              s += A[i*K + p] * B[p*N + j];
            }
            C[i*N + j] = s;
          }
        }
      }
    }
  }
}

int main(int argc, char** argv){
  int M = 1024, N = 1024, K = 1024;
  if (argc == 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }

  std::vector<double> A((size_t)M*K), B((size_t)K*N), C((size_t)M*N, 0.0);

  // Initialize with deterministic data
  for (int i=0;i<M;i++) for (int k=0;k<K;k++) A[i*K+k] = (i+k)%7 - 3;
  for (int k=0;k<K;k++) for (int j=0;j<N;j++) B[k*N+j] = (k-j)%5 + 0.5;

  auto t0 = std::chrono::high_resolution_clock::now();
  gemm_blocked_omp(A.data(), B.data(), C.data(), M, N, K, /*BS=*/128);
  auto t1 = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count();
  double gflops = 2.0 * M * (double)N * K / 1e9 / secs;
  std::cout << "Time: " << secs << " s, ~" << gflops << " GFLOP/s\n";

  // Print a quick checksum so you know it ran
  double sum=0; for (double x: C) sum += x;
  std::cout << "Checksum: " << std::setprecision(17) << sum << "\n";
  return 0;
}
