#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

template <class T>
class dev_array
{
public:
	explicit dev_array() : Start(0), Size(0) {}

	explicit dev_array(size_t size) {
		allocate(size);
	}

	// Constructs from a host array. Allocates device memory then calls copy from. 
	dev_array(const T* host_src, size_t size) {
		allocate(size);
		copy_from(host_src, size);
	}

	~dev_array() {
		free();
	}

	size_t size() {
		return Size;
	}

	size_t size() const {
		return Size;
	}

	/*
	* Copies data from host to device.
	* Parameters:
	* host_src - pointer to the source array
	* size - size of the data copied over
	* Throws: std::length_error, std::runtime_error
	*/
	void copy_from(const T* host_src, size_t size) {
		if (size > Size) {
			throw std::length_error("dev_array: host size exceeds device size");
		}
		auto result = cudaMemcpy(Start, host_src, size * sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			throw std::runtime_error("dev_array: failed to copy memory from host to device");
		}
	}

	/*
	* Copies data from device to host.
	* Parameters:
	* host_dest - pointer to the destination array
	* size - size of the data copied over
	* Throws: std::length_error, std::runtime_error
	*/
	void copy_to(T* host_dest, size_t size) {
		if (Size > size) {
			throw std::length_error("dev_array: device size exceeds host size");
		}
		auto result = cudaMemcpy(host_dest, Start, size * sizeof(T), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			throw std::runtime_error("dev_array: failed to copy memory from device to host");
		}
	}

	T* dev_ptr() {
		return Start;
	}

	const T* dev_ptr() const {
		return Start;
	}

	void resize(size_t size) {
		free();
		allocate(size);
	}

private:
	T* Start;
	size_t Size;

	void allocate(size_t size) {
		Size = size;
		size_t byte_size = Size * sizeof(T);
		auto result = cudaMalloc((void**)&Start, byte_size);
		if (result != cudaSuccess) {
			Start = nullptr;
			throw std::runtime_error("dev_array: failed to allocate device memory");
		}
	}

	void free() {
		if (Start != 0) {
			cudaFree(Start);
			Start = 0;
			Size = 0;
		}
	}

};