#include <math.h>
#include <time.h>
#include <vector>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include "ops.h"
#include "dev_array.h"

// Adds two vectors. Requires: v1.size() == v2.size().
std::vector<float> sum(const std::vector<float>& v1, const std::vector<float>& v2) {
	size_t n = v1.size();
	auto h_result = std::vector<float>(n);
	auto d_result = dev_array<float>(&v1[0], n);
	auto d_summand = dev_array<float>(&v2[0], n);
	add_vectors(d_result.dev_ptr(), d_summand.dev_ptr(), n);
	d_result.copy_to(&h_result[0], n);
	return h_result;
}

//int main() {
//	std::vector<float> a = { 1,2,3,4,5 };
//	std::vector<float> b = { 1,2,3,4,5 };
//	auto s = sum(a, b);
//	for (auto elem : s) {
//		std::cout << elem << std::endl;
//	}
//	std::cin;
//}