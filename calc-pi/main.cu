#include <iostream>

using namespace std;

const int N = 1 << 20;
const double STEP = 1.0 / N;

__global__
void kernel(double * result)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    double point = i * STEP;
    double f = sqrt(1 - (point * point));

    result[i] = f;
}

int main()
{
    double * R = new double[N];
    
    double * device_R = nullptr;
    auto bytes = N * sizeof(double);
    cudaMalloc(&device_R, bytes);

    cudaMemcpy(device_R, R, bytes, cudaMemcpyHostToDevice);

    int num_threads = 1 << 5;
    int num_blocks = N / num_threads;
    dim3 threads(num_threads);
    dim3 blocks(num_blocks);

    // time start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<blocks, threads>>>(device_R);

    // time stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Time: " << milliseconds << "ms" << endl;

    cudaMemcpy(R, device_R, bytes, cudaMemcpyDeviceToHost);

    double sum_fi = 0;
    for (int i = 1; i < N - 1; i++) {
        sum_fi += R[i];
    }
    double pi = 4 * (STEP * (0.5 + sum_fi));
    cout << "Pi = " << pi;

    return 0;
}
