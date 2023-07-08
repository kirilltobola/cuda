#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cstdio>
#include <cstdlib>

void usage(const char* filename)
{
	printf("Calculating a saxpy transform for two random vectors of the given length.\n");
	printf("Usage: %s <n>\n", filename);
}

using namespace thrust;
//using namespace thrust::placeholders;

// TODO: Please refer to sorting examples:
// http://code.google.com/p/thrust/
// http://code.google.com/p/thrust/wiki/QuickStartGuide#Transformations

struct saxpy
{
	float a;
	// Constructor:
	saxpy(float a): a(a) {}

	// TODO: define operator ()
	__host__ __device__ float operator () (float x, float y) {
	    return a * x + y;
	}
};

int main(int argc, char* argv[])
{
	const int printable_n = 128;

	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	if (n <= 0)
	{
		usage(argv[0]);
		return 0;
	}
	//cudaSetDevice(2);
	
	// TODO: Generate 3 vectors on host ( z = a * x + y)
	thrust::host_vector<float> x(n);
	thrust::host_vector<float> y(n);
	for (int i = 0; i < n; i++) {
	    x[i] = rand();
	    y[i] = rand();
	}
	thrust::host_vector<float> z(n);


	// Print out the input data if n is small.
	if (n <= printable_n) {
		printf("Input data:\n");
		for (int i = 0; i < n; i++)
			printf("%f   %f\n", 1.f*x[i] / RAND_MAX, 1.f*y[i] / RAND_MAX);
		printf("\n");
	}

	// TODO: Transfer data to the device.
	thrust::device_vector<float> device_x = x;
	thrust::device_vector<float> device_y = y;
	thrust::device_vector<float> device_z(n);

	float a = 2.5f;
	// TODO: Use transform to make an saxpy operation
    thrust::transform(
        device_x.begin(), device_x.end(),
        device_y.begin(),
        device_z.begin(),
        saxpy(a)
    );
	
	// Note: you may use placeholders


	// TODO: Transfer data back to host.
    z = device_z;

	// Print out the output data if n is small.
	if (n <= printable_n)
	{
		printf("Output data:\n");
		for (int i = 0; i < n; i++)
			printf("%f * %f + %f = %f\n", a, 1.f*x[i] / RAND_MAX, 1.f*y[i] / RAND_MAX, z[i] / RAND_MAX);
		printf("\n");
	}

	return 0;
}

