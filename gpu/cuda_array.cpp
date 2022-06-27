#include "cuda_array.h"
#include <cuda_runtime.h>
#include <stdexcept>

template <class T>
class cuda_array
{
public:
	explicit cuda_array() : Start(0), Size(0) {}

	explicit cuda_array(size_t size) {
		allocate(size);
	}

	// Constructs from a host array
	cuda_array(const T* host_src, size_t size) {
		allocate(size);
		from(host_src, size);
	}

	~cuda_array() {
		free();
	}

	size_t size() {
		return Size;
	}

	/*
	* Copies data from host to device.
	* Parameters:
	* host_src - pointer to the source array
	* size - size of the data copied over
	* Throws: std::length_error, std::runtime_error
	*/
	void from(const T* host_src, size_t size) {
		if (size > Size) {
			throw std::length_error("cuda_array: host size exceeds device size");
		}
		auto result = cudaMemcpy(Start, host_src, size * sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error("cuda_array: failed to copy memory from host to device");
		}
	}

	/*
	* Copies data from device to host.
	* Parameters:
	* host_dest - pointer to the destination array
	* size - size of the data copied over
	* Throws: std::length_error, std::runtime_error
	*/
	void out_to(T* host_dest, size_t size) {
		if (Size > size) {
			throw std::length_error("cuda_array: device size exceeds host size");
		}
		auto result = cudaMemcpy(host_dest, Start, size * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error("cuda_array: failed to copy memory from device to host");
		}
	}

	T* ptr() {
		return Start;
	}

	void resize(size_t size) {
		free();
		allocate(size);
	}

private:
	T *Start;
	size_t Size;

	void allocate(size_t size) {
		Size = size;
		size_t byte_size = Size * sizeof(T);
		auto result = cudaMalloc((void**)&Start, byte_size);
		if (result != cudaSuccess) {
			Start = nullptr;
			throw std::runtime_error("cuda_array: failed to allocate device memory");
		}
	}

	void free() {
		Size = 0;
		auto result = cudaFree(Start);
		if (result != cudaSuccess) {
			throw std::runtime_error("cuda_array: failed to free device memory");
		}
	}

};