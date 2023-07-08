#include <iostream>

using namespace std;

#define N 1024


__global__
void kernel(const int * a, const int * b, int * c)
{
    int row = (blockDim.x * blockIdx.x) + threadIdx.x;
    int col = (blockDim.y * blockIdx.y) + threadIdx.y;

    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
}

int main()
{
    // host
    int * a = new int[N * N];
    for (int i = 0; i < N * N; i++) {
        a[i] = 1;
    }
    
    int * b = new int[N * N];
    for (int i = 0; i < N * N; i++) {
        b[i] = -8;
    }
    
    int * c = new int[N * N];

    //device
    int * device_a = nullptr;
    cudaMalloc(&device_a, N * N * sizeof(int));
    cudaMemcpy(device_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);

    int * device_b = nullptr;
    cudaMalloc(&device_b, N * N * sizeof(int));
    cudaMemcpy(device_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    int * device_c = nullptr;
    cudaMalloc(&device_c, N * N * sizeof(int));
    cudaMemcpy(device_c, c, N * N * sizeof(int), cudaMemcpyHostToDevice);

    int num_threads = 1 << 2;
    int num_blocks = N / num_threads;
    dim3 threads(num_threads, num_threads);
    dim3 blocks(num_blocks, num_blocks);

    // time start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<blocks, threads>>>(device_a, device_b, device_c);

    // time stop
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "N = " << N << endl;
    cout << "Time: " << milliseconds << "ms" << endl;
    
    cudaMemcpy(c, device_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    if (false) {
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++) {
                cout << c[i * N + j] << " ";
            }
            cout << endl;
        }  
    }

    return 0;
}
