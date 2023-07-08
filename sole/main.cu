#include <bits/stdc++.h>

using namespace std;

#define N 2048

__global__
void kernel(int * A, int * b, double * x0, double * x1)
{
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    double sum = 0;
    for (int i = 0; i < N; i++) {
        if (idx != i)    
            sum += A[N * idx + i] * x0[i];
    }

    double alpha = 1.0 / A[N * idx + idx];
    x1[idx] = alpha * (b[idx] - sum);
}

int main()
{
    // host
    int * A = new int[N * N];
    for (int i = 0; i < N; i++) {
        int sum_row = 0;
        for (int j = 0; j < N; j++) {
            if (i != j) {
                A[i * N + j] = rand() & ((1 << 10) - 1); // & 1024
                sum_row += A[i * N + j];
            }
        }
        A[i * N + i] = sum_row + 1024; // diagonal must be > sum(row)
    }

    int * b = new int[N];
    for (int i = 0; i < N; i++) {
        b[i] = rand() & ((1 << 12) - 1);
    }

    double * x0 = new double[N];
    for (int i = 0; i < N; i++) {
        x0[i] = rand() & ((1 << 4) - 1);
    }

    double * x1 = new double[N];

    // device
    int * device_A = nullptr;
    cudaMalloc(&device_A, N * N * sizeof(int));
    cudaMemcpy(device_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);


    int * device_b = nullptr;
    cudaMalloc(&device_b, N * sizeof(int));
    cudaMemcpy(device_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    double * device_x0 = nullptr;
    cudaMalloc(&device_x0, N * sizeof(double));
    cudaMemcpy(device_x0, x0, N * sizeof(double), cudaMemcpyHostToDevice);

    double * device_x1 = nullptr;
    cudaMalloc(&device_x1, N * sizeof(double));

    // kernel config
    int num_threads = 1 << 5;
    int num_blocks = N / num_threads;
    dim3 threads(num_threads);
    dim3 blocks(num_blocks);


    float epsilon = 0.000001;
    bool descending = true;

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    while (descending) {
        descending = false;
        kernel<<<blocks, threads>>>(device_A, device_b, device_x0, device_x1);
        cudaDeviceSynchronize();

        cudaMemcpy(x1, device_x1, N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; i++) {
            if (abs(x0[i] - x1[i]) > epsilon) {
                descending = true;
                break;
            }
        }

        swap(x0, x1);
        swap(device_x0, device_x1);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "N = 2048" << endl;
    cout << "Time: " << milliseconds << "ms" << endl;
    
    return 0;
}
