#pragma once

void matmul_naive(const float* A, const float* B, float* C, int N);
void fill_matrix(float* M, int N, float value = 1.0f);
bool matrices_equal(const float* A, const float* B, int N, float epsilon = 1e-4f);


