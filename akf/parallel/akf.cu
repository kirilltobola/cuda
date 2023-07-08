#include <iostream>
#include <algorithm>
#include <climits>

using namespace std;

#define N 60
#define SEP ';'

__global__
void kernel(int bits_of_signal, unsigned long long * res)
{
    // хранит максимальное значение akf для
    // определенного числа (blockIdx.x).
    __shared__ int max_akf;
    max_akf = 0;
    
    __syncthreads();

    __shared__ int signal[N];

    // генерация битового представления числа
    int bit = (blockIdx.x >> threadIdx.x) & 1;
    if (bit == 0) {
        bit = -1;  
    }
    signal[bits_of_signal-1 - threadIdx.x] = bit;

    __syncthreads();

    // рассчет akf для i-го сдвига
    int i = threadIdx.x;
    int akf = 0;
    for (int j = 0; j < bits_of_signal; j++) {
        if (i + j < bits_of_signal) {
            akf += signal[i + j] * signal[j];
        }
    }

    // для всех сдвигов, кроме 0-го
    // находим значение максимального akf
    // и сохраняем в переменную max_akf
    if (threadIdx.x > 0) {
        atomicMax(&max_akf, abs(akf));
    }
    __syncthreads();

    // максимальный akf и форму сигнала
    // будем хранить в одной переменной:
    // сохраняем максимальный akf.
    unsigned long long _res = max_akf;

    // сдвигаем влево на длину сигнала
    // максимальный akf.
    _res <<= bits_of_signal;
    
    // помещаем форму сигнала в конец числа.
    // в итоге получается число, в котором:
    // 00...0{max_akf}{signal};
    _res |= blockIdx.x;

    // из всех сигналов длины bits_of_signal
    // находим с минимальным максимальным akf
    // и сохраняем в переменную res;
    atomicMin(
        res,
        _res
    );    
}

string get_binary_repr(int signal, int bits_of_signal);

int main()
{
    cout << "bits" << SEP
        << "max_leaf" << SEP
        << "signal_bin" << SEP
        << "ms" << SEP << endl;
    
    for (int bits_of_signal = 6; bits_of_signal < 30; bits_of_signal++) {
        cout << bits_of_signal << SEP;

        dim3 num_threads(bits_of_signal);
        dim3 num_blocks(1 << bits_of_signal);

        // host
        unsigned long long res = ULLONG_MAX;

        // device
        unsigned long long * device_res = nullptr;

        auto bytes = sizeof(unsigned long long);

        cudaMalloc(&device_res, bytes);
        cudaMemcpy(device_res, &res, bytes, cudaMemcpyHostToDevice);

        // time start
        cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

        kernel<<<num_blocks, num_threads>>>(bits_of_signal, device_res);
        cudaDeviceSynchronize();

        // time stop
        cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

        cudaMemcpy(&res, device_res, bytes, cudaMemcpyDeviceToHost);

        // в res содержится 00..0{akf}{signal};
        int signal = res & ((1 << bits_of_signal) - 1);  
        int akf = res >> bits_of_signal;

        string signal_bin = get_binary_repr(signal, bits_of_signal);
        
        reverse(signal_bin.begin(), signal_bin.end());
        cout << akf << SEP << signal_bin << SEP << milliseconds << "ms" << endl;
    }
    return 0;
}

string get_binary_repr(int signal, int bits_of_signal)
{
    int n = signal;
    string n_binary = "";
    for (int i = 0; i < bits_of_signal; i++) {
        n_binary += '0' + n % 2;
        n /= 2;
    }
    return n_binary;
}
