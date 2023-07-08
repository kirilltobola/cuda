#include <iostream>

using namespace std;

#define N 2048
#define M 2048

__global__
void kernel(const int * a, const int * b, int * c)
{
    int idx = M * (blockDim.x * blockIdx.x + threadIdx.x) + 
        (blockDim.y * blockIdx.y + threadIdx.y);
    c[idx] = a[idx] - b[idx];
}

int main()
{
    int * A = new int[N * M];
    int * B = new int[N * M];
    int * C = new int[N * M];
    for (int i = 0; i < N * M; i++) {
        A[i] = 1;
        B[i] = 15;
    }
    
    // allocate memory on device
    auto bytes = (N * M) * sizeof(int);
    int * device_A = nullptr;
    int * device_B = nullptr;
    int * device_C = nullptr;
    cudaMalloc(&device_A, bytes);
    cudaMalloc(&device_B, bytes);
    cudaMalloc(&device_C, bytes);

    // copy matricies host -> device
    cudaMemcpy(device_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, C, bytes, cudaMemcpyHostToDevice);

    int num_threads = 16;
    int num_blocks = (N) / num_threads;
    dim3 threads(num_threads, num_threads);
    dim3 blocks(num_blocks, num_blocks);
    
    //dim3 threads(4, 4);
    //dim3 blocks(N / 4, M / 4);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //call kernel
    kernel<<<blocks, threads>>>(device_A, device_B, device_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "rows = " << N << "cols = " << M << endl;
    cout << "Time:" << milliseconds << "ms" << endl;

    // copy result device -> host
    cudaMemcpy(C, device_C, bytes, cudaMemcpyDeviceToHost);

    //for (int i = 0; i < N; i++) {
    //    for (int j = 0; j < M; j++) {
    //        cout << C[i * M + j];   
    //    }
    //    cout << endl;
    //}

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
        
    return 0;
}
