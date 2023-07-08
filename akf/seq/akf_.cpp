#include <iostream>
#include <string>
#include <algorithm>
#include <chrono>


using namespace std;

int _get_akf(int signal, int n);
int calc_max(int signal, int bits);
void find_leaf(int bits_of_signal);

char SEP = ';';
bool CSV_PRINT = true;

int main()
{
	if (CSV_PRINT) {
		cout << "bits" << SEP
	    << "max_leaf" << SEP
	    << "signal_bin" << SEP
	    << "ms" << SEP << endl;
	}
    
    for (int n = 5; n < 30; n++) {
        auto _start = chrono::steady_clock::now();

        find_leaf(n);

        auto _end = chrono::steady_clock::now();
        auto elapsed = _end - _start;
        if (CSV_PRINT) {
        	cout << chrono::duration_cast< chrono::milliseconds > (elapsed).count() << SEP;
    		cout << endl;	
        }
    }
    return 0;
}

void find_leaf(int bits_of_signal)
{
	if (CSV_PRINT) cout << bits_of_signal << SEP;
    // best signals start with 3 ones.
    int numbers_to_check = 1 << (bits_of_signal - 3);
    // int * signal = new int[bits_of_signal];
	int signal;

    int n, min_max_leaf;
    for (int num = 0; num < numbers_to_check; num++) {
		signal = 0;
		signal |= 1 << (bits_of_signal - 1);
		signal |= 1 << (bits_of_signal - 2);
		signal |= 1 << (bits_of_signal - 3);
		signal += num;

		// int max_leaf = calc_max(signal, bits_of_signal);
        int max_leaf = _get_akf(signal, bits_of_signal);

        // cout << "max leaf=" << max_leaf << endl;

        if (num == 0) {
            n = num;
            min_max_leaf = max_leaf;
            continue;
        }
        if (min_max_leaf >= max_leaf) {
            n = num;
            min_max_leaf = max_leaf;
        }
    }
    if (CSV_PRINT) cout << min_max_leaf << SEP; 

    // print binary
    if (true) {
	    string n_binary = "";
	    for (int i = 0; i < bits_of_signal - 3; i++) {
	        n_binary += '0' + n % 2;
	        n /= 2;
	    }
	    reverse(n_binary.begin(), n_binary.end());
	    cout << "111" + n_binary << SEP;
	}
}

int _get_akf(int signal, int bits)
{
	int akf = -1;
	int mask = (1 << bits) - 1;
	int n = bits;

	for (int i = 1; i < n; i++) {
		bits--;
		int signal_shr = (signal >> i);		
		int _xor = signal ^ signal_shr;

		_xor &= mask >> i;

		// count ones
		int bits_diff = __builtin_popcount(_xor);
        int _akf = bits - 2 * bits_diff;
		
		int leaf = abs(_akf);
		if (akf <= leaf) {
			akf = leaf;
		}
	}
	return akf;
}
