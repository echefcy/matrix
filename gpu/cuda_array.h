#pragma once

template<class T>
class cuda_array
{
public:
	cuda_array();
	cuda_array(size_t);
	cuda_array(const T*, size_t);
	~cuda_array();
	size_t size();
	void from(const T*, size_t);
	void out_to(T*, size_t);
	T* ptr();
	void resize(size_t);
	void resize(size_t);
};

