#include <iostream>

using namespace std;

int main()
{
    cudaDeviceProp dp;
    cudaGetDeviceProperties(
        &dp,
        0
    );

    cout << "name: " << dp.name << endl;
    cout << "total memory: " << dp.totalGlobalMem << endl;
    cout << "shared memory per block: " << dp.sharedMemPerBlock << endl;
    cout << "max threads per block: " << dp.maxThreadsPerBlock << endl;
    cout << "max grid size: " << dp.maxGridSize << endl;
    
    return 0;
}
