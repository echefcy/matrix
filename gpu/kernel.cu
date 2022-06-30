#include <algorithm>
#include "cuda_runtime.h"
#include "kernel.h"

static const unsigned int max_threads_per_block = 512;

// = ceil(a/b)
int div_ceil(unsigned int a, unsigned int b) {
	return a % b == 0 ? a / b : a / b + 1;
}

void sum(float* result, const float* summand, unsigned int n) {
	// newer graphics cards can go up to 1024 threads per block https://en.wikipedia.org/wiki/CUDA
	int threads_per_block = std::min(n, max_threads_per_block);
	int blocks_per_grid = div_ceil(n, max_threads_per_block);
	kernel_sum<<<blocks_per_grid, threads_per_block>>>(result, summand, n);
}

__global__ void kernel_sum(float* result, const float* summand, unsigned int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) result[i] += summand[i];
}