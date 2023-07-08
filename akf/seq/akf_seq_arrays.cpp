#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>

using namespace std;


int * get_akf(int * signal, int n);
void find_leaf(int bits_of_signal);

char SEP = ';';

int main()
{
    cout << "bits" << SEP
    << "max_leaf" << SEP
    << "signal_bin" << SEP
    << "ms" << SEP << endl;

    for (int n = 5; n < 30; n++) {
        auto _start = chrono::steady_clock::now();

        find_leaf(n);

        auto _end = chrono::steady_clock::now();
        auto elapsed = _end - _start;
        cout << chrono::duration_cast< chrono::milliseconds > (elapsed).count() << SEP;
        cout << endl;
    }

    return 0;
}

void find_leaf(int bits_of_signal)
{
    cout << bits_of_signal << SEP;
    int numbers_to_check = 1 << (bits_of_signal - 3);
    int * signal = new int[bits_of_signal];

    int n, min_max_leaf;
    for (int num = 0; num < numbers_to_check; num++) {
        signal[0] = 1;
        signal[1] = 1;
        signal[2] = 1;
        for (int bit_pos = 0; bit_pos < bits_of_signal - 3; bit_pos++) {
            int bit = (num >> bit_pos) & 1;
            if (bit == 0) bit = -1;
            signal[bits_of_signal-1 - bit_pos] = bit;
        }

        int * akf = get_akf(signal, bits_of_signal);
        int max_leaf = abs(akf[1]);
        for (int i = 1; i < bits_of_signal; i++) {
            if (max_leaf < abs(akf[i])) {
               max_leaf = abs(akf[i]);
            }
        }

        if (num == 0) {
            n = num;
            min_max_leaf = max_leaf;
            continue;
        }
        if (min_max_leaf >= max_leaf) {
            n = num;
            min_max_leaf = max_leaf;
        }
        delete[] akf;
    }
    cout << min_max_leaf << SEP;

    delete[] signal;

    // print binary
    string n_binary = "";
    for (int i = 0; i < bits_of_signal - 3; i++) {
        n_binary += '0' + n % 2;
        n /= 2;
    }
    reverse(n_binary.begin(), n_binary.end());
    cout << "111" + n_binary << SEP;
}

int * get_akf(int * signal, int n)
{
    int * akf = new int[n];

    for (int i = 0; i < n; i++) {
        akf[i] = 0;
        for (int j = 0; j < n; j++) {
            if (i + j < n) {
                akf[i] += signal[i + j] * signal[j];
            }
        }
    }
    return akf;
}
