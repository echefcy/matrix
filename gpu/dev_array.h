#pragma once

template<class T>
class dev_array
{
public:
	dev_array();

	dev_array(size_t);

	// Constructs from a host array. Allocates device memory then calls copy from. 
	dev_array(const T*, size_t);

	~dev_array();

	size_t size();

	/*
	* Copies data from host to device.
	* Parameters:
	* host_src - pointer to the source array
	* size - size of the data copied over
	* Throws: std::length_error, std::runtime_error
	*/
	void copy_from(const T*, size_t);

	/*
	* Copies data from device to host.
	* Parameters:
	* host_dest - pointer to the destination array
	* size - size of the data copied over
	* Throws: std::length_error, std::runtime_error
	*/
	void copy_to(T*, size_t);

	T* dev_ptr();
	
	const T* dev_ptr() const;

	void resize(size_t);
};

