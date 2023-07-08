#include <iostream>

using namespace std;

#define N 1024
#define SHARED_MEMORY_SIZE 16


__global__
void kernel(const int * a, const int * b, int * c)
{
    __shared__ int sh_a[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];
    __shared__ int sh_b[SHARED_MEMORY_SIZE][SHARED_MEMORY_SIZE];

    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    int sum = 0;
    for (int i = 0; i < (N / SHARED_MEMORY_SIZE); i++) {
        // load data to shared memory
        auto global_row = row * N;
        auto columns_i = i * SHARED_MEMORY_SIZE;
        auto col_i = threadIdx.x;
        sh_a[threadIdx.y][threadIdx.x] = a[
            global_row + (columns_i + col_i)
        ];

        auto global_col = col;
        auto rows_i = i * SHARED_MEMORY_SIZE * N;
        auto row_i = threadIdx.y * N;
        sh_b[threadIdx.y][threadIdx.x] = b[
             global_col + (rows_i + row_i)
        ];
        __syncthreads();

        // calculate
        for (int j = 0; j < SHARED_MEMORY_SIZE; j++) {
            sum += sh_a[threadIdx.y][j] * sh_b[j][threadIdx.x];
        }
        __syncthreads();
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
        b[i] = -1;
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

    int num_threads = SHARED_MEMORY_SIZE;
    int num_blocks = N / num_threads;
    dim3 threads(num_threads, num_threads);
    dim3 blocks(num_blocks, num_blocks);

    // time start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<blocks, threads>>>(device_a, device_b, device_c);
    cudaDeviceSynchronize();

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
