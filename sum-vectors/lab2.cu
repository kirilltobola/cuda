
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <thread>

#include <chrono>

#define N 10000000

__global__ void addKernel(int * c, const int * a, const int * b)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void add(int * c, const int * a, const int * b)
{
	for (int i = 0; i < N; i++) {
		c[i] = a[i] + b[i];
	}
}

void call_add(int * c, const int * a, const int * b)
{
	auto s = std::chrono::steady_clock::now();
	add(c, a, b);
	auto e = std::chrono::steady_clock::now();

	std::cout << "CPU elapsed time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
		<< " ms" << std::endl;

	// Print results
	int sum_ = 0;
	for (int i = 0; i < N; i++) {
		sum_ += c[i];
	}
	std::cout << "sum = " << sum_ << std::endl;
}

int main()
{
	int * vec1 = new int[N];
	int * vec2 = new int[N];
	int * res = new int[N];

	// Init vectors
	for (int i = 0; i < N; i++) {
		vec1[i] = 1;
		vec2[i] = 0;
	}

	// CPU
	call_add(res, vec1, vec2);

	// Init vectors
	for (int i = 0; i < N; i++) {
		vec1[i] = 1;
		vec2[i] = 0;
	}

	// Alloc memory on gpu
	int * dev_vec1 = nullptr;
	int * dev_vec2 = nullptr;
	int * dev_res = nullptr;
	cudaMalloc((void**)&dev_vec1, N * sizeof(int));
	cudaMalloc((void**)&dev_vec2, N * sizeof(int));
	cudaMalloc((void**)&dev_res, N * sizeof(int));

	// From Host to Device
	cudaMemcpy(dev_vec1, vec1, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec2, vec2, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res, res, N * sizeof(int), cudaMemcpyHostToDevice);

	// Measure time
	float time;
	cudaEvent_t	start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Launch a kernel on the GPU with one block; N threads.
	addKernel <<<1, N>>> (dev_res, dev_vec1, dev_vec2);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("GPU elapsed time: %.5f ms \n", time);

	// From Device to Host
	cudaMemcpy(res, dev_res, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Print results
	int sum = 0;
	for (int i = 0; i < N; i++) {
		sum += res[i];
		// std::cout << res[i] << std::endl;
	}
	std::cout << "sum = " << sum << std::endl;

	cudaFree(dev_vec1);
	cudaFree(dev_vec2);
	cudaFree(dev_res);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();

    return 0;
}
