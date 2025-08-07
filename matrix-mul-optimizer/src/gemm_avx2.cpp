// From-scratch high-performance GEMM (C += A * B) in double precision.
// - Cache blocking (MC/NC/KC)
// - Packing of A (row panels) and B (column panels)
// - AVX2+FMA 6x8 micro-kernel
// - OpenMP threading across macro tiles
// Assumes row-major A[MxK], B[KxN], C[MxN].
//
// Build: g++ -O3 -march=native -mavx2 -mfma -fopenmp gemm_avx2.cpp -o gemm_avx2

#include <immintrin.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#ifdef _OPENMP
  #include <omp.h>
#endif

// Tunables (good starting points for modern desktop/server CPUs)
constexpr int MC = 288;   // rows of A per macro tile (multiple of MR)
constexpr int NC = 512;   // cols of B per macro tile (multiple of NR)
constexpr int KC = 256;   // shared-k depth per panel
constexpr int MR = 6;     // micro-k rows
constexpr int NR = 8;     // micro-k cols

static inline int ceil_div(int a, int b){ return (a + b - 1)/b; }

// Pack A block (mc x kc) from row-major into [mc rounded to MR] x kc, row-by-row
static void pack_A(const double* A, int lda, double* Ap, int mc, int kc){
  // Layout: blocks of MR rows, each row has kc contiguous doubles
  for (int i = 0; i < mc; i += MR){
    int r = std::min(MR, mc - i);
    for (int rr = 0; rr < MR; ++rr){
      const double* src = (rr < r) ? (A + (i+rr)*lda) : nullptr;
      for (int p = 0; p < kc; ++p){
        Ap[rr*kc + p] = src ? src[p] : 0.0;
      }
    }
    Ap += MR * kc;
  }
}

// Pack B panel (kc x nc) from row-major into kc * [nc rounded to NR] with NR-interleaving
static void pack_B(const double* B, int ldb, double* Bp, int kc, int nc){
  // Layout: for p in [0..kc): store a vector of NR columns per step
  for (int j = 0; j < nc; j += NR){
    int c = std::min(NR, nc - j);
    for (int p = 0; p < kc; ++p){
      const double* src = B + p*ldb + j;
      for (int cc = 0; cc < NR; ++cc){
        Bp[p*NR + cc] = (cc < c) ? src[cc] : 0.0;
      }
    }
    Bp += kc * NR;
  }
}

// 6x8 AVX2 micro-kernel: C[6x8] += A_pack[6 x kc] * B_pack[kc x 8]
// A_pack: 6 rows, each 'kc' contiguous; B_pack: for each p, 8 contiguous values.
static inline void microkernel_6x8(int kc,
                                   const double* __restrict Ap,
                                   const double* __restrict Bp,
                                   double* __restrict C, int ldc)
{
  __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
  __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
  __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
  __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
  __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
  __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

  for (int p = 0; p < kc; ++p){
    // B is laid out as consecutive 8 doubles per p
    const double* bp = Bp + p*NR;
    __m256d b0 = _mm256_loadu_pd(bp + 0);
    __m256d b1 = _mm256_loadu_pd(bp + 4);
    __m256d a0 = _mm256_set1_pd(Ap[0*kc + p]);
    __m256d a1 = _mm256_set1_pd(Ap[1*kc + p]);
    __m256d a2 = _mm256_set1_pd(Ap[2*kc + p]);
    __m256d a3 = _mm256_set1_pd(Ap[3*kc + p]);
    __m256d a4 = _mm256_set1_pd(Ap[4*kc + p]);
    __m256d a5 = _mm256_set1_pd(Ap[5*kc + p]);

    c00 = _mm256_fmadd_pd(a0, b0, c00); c01 = _mm256_fmadd_pd(a0, b1, c01);
    c10 = _mm256_fmadd_pd(a1, b0, c10); c11 = _mm256_fmadd_pd(a1, b1, c11);
    c20 = _mm256_fmadd_pd(a2, b0, c20); c21 = _mm256_fmadd_pd(a2, b1, c21);
    c30 = _mm256_fmadd_pd(a3, b0, c30); c31 = _mm256_fmadd_pd(a3, b1, c31);
    c40 = _mm256_fmadd_pd(a4, b0, c40); c41 = _mm256_fmadd_pd(a4, b1, c41);
    c50 = _mm256_fmadd_pd(a5, b0, c50); c51 = _mm256_fmadd_pd(a5, b1, c51);
  }

  // C is row-major with leading dim ldc
  _mm256_storeu_pd(C + 0*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 0*ldc + 0), c00));
  _mm256_storeu_pd(C + 0*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 0*ldc + 4), c01));

  _mm256_storeu_pd(C + 1*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 1*ldc + 0), c10));
  _mm256_storeu_pd(C + 1*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 1*ldc + 4), c11));

  _mm256_storeu_pd(C + 2*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 2*ldc + 0), c20));
  _mm256_storeu_pd(C + 2*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 2*ldc + 4), c21));

  _mm256_storeu_pd(C + 3*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 3*ldc + 0), c30));
  _mm256_storeu_pd(C + 3*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 3*ldc + 4), c31));

  _mm256_storeu_pd(C + 4*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 4*ldc + 0), c40));
  _mm256_storeu_pd(C + 4*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 4*ldc + 4), c41));

  _mm256_storeu_pd(C + 5*ldc + 0, _mm256_add_pd(_mm256_loadu_pd(C + 5*ldc + 0), c50));
  _mm256_storeu_pd(C + 5*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + 5*ldc + 4), c51));
}

// Scalar tail update for partial tiles: C[ri x rj] += A_pack[ri x kc] * B_pack[kc x rj]
static void tail_update(int ri, int rj, int kc,
                        const double* Ap, const double* Bp,
                        double* C, int ldc)
{
  for (int i = 0; i < ri; ++i){
    for (int j = 0; j < rj; ++j){
      double s = 0.0;
      for (int p = 0; p < kc; ++p){
        s += Ap[i*kc + p] * Bp[p*NR + j]; // Bp laid out NR-wide, take first rj cols
      }
      C[i*ldc + j] += s;
    }
  }
}

// Top-level GEMM: C += A * B
void gemm_opt(int M, int N, int K,
              const double* A, int lda,
              const double* B, int ldb,
              double* C, int ldc)
{
  std::vector<double> Bpanel; Bpanel.reserve((size_t)KC * NR * ceil_div(N, NR));
  // Outer blocking over N and K
  for (int jc = 0; jc < N; jc += NC){
    int nc = std::min(NC, N - jc);
    for (int pc = 0; pc < K; pc += KC){
      int kc = std::min(KC, K - pc);

      // Pack B panel once per (pc, jc)
      Bpanel.assign((size_t)ceil_div(nc, NR) * kc * NR, 0.0);
      {
        double* Bp = Bpanel.data();
        const double* Bblk = B + pc*ldb + jc;
        pack_B(Bblk, ldb, Bp, kc, nc);
      }

      // Parallelize A-side macro tiles
      #pragma omp parallel
      {
        std::vector<double> Apanel; Apanel.reserve((size_t)MC * kc);
        #pragma omp for schedule(static)
        for (int ic = 0; ic < M; ic += MC){
          int mc = std::min(MC, M - ic);

          // Pack A block for this thread
          Apanel.assign((size_t)ceil_div(mc, MR) * MR * kc, 0.0);
          {
            const double* Ablk = A + ic*lda + pc;
            pack_A(Ablk, lda, Apanel.data(), mc, kc);
          }

          // Compute C block
          const double* Bp_base = Bpanel.data();
          for (int j = 0; j < nc; j += NR){
            int jb = std::min(NR, nc - j);
            const double* Bp = Bp_base + (size_t)(j/NR) * kc * NR;

            for (int i = 0; i < mc; i += MR){
              int ib = std::min(MR, mc - i);
              double* Cblk = C + (ic + i)*ldc + (jc + j);
              const double* Ap = Apanel.data() + (size_t)(i/MR) * MR * kc;

              if (ib == MR && jb == NR){
                microkernel_6x8(kc, Ap, Bp, Cblk, ldc);
              } else {
                // Partial tile via scalar tail
                tail_update(ib, jb, kc, Ap, Bp, Cblk, ldc);
              }
            }
          }
        }
      } // omp parallel
    }
  }
}

// --- Demo / CLI driver ---
int main(int argc, char** argv){
  int M = 2048, N = 2048, K = 2048;
  if (argc == 4){ M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }

  std::vector<double> A((size_t)M*K), B((size_t)K*N), C((size_t)M*N, 0.0);

  // Deterministic initialization
  for (int i=0;i<M;i++) for (int k=0;k<K;k++) A[(size_t)i*K + k] = ( (i*131 + k*7) % 17 ) - 8;
  for (int k=0;k<K;k++) for (int j=0;j<N;j++) B[(size_t)k*N + j] = ( (k*19 - j*3) % 13 ) + 0.25;

  auto t0 = std::chrono::high_resolution_clock::now();
  gemm_opt(M, N, K, A.data(), K, B.data(), N, C.data(), N);
  auto t1 = std::chrono::high_resolution_clock::now();

  double secs = std::chrono::duration<double>(t1 - t0).count();
  double gflops = 2.0 * (double)M * N * K / 1e9 / secs;
  double checksum = 0.0; for (double x : C) checksum += x;

  std::printf("Time: %.6f s  ~%.2f GFLOP/s\n", secs, gflops);
  std::printf("Checksum: %.17g\n", checksum);
  return 0;
}
