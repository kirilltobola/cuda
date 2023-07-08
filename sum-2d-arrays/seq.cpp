#include <iostream>
#include <chrono>

using namespace std;

#define ROWS 2048
#define COLS 2048

int main()
{
    int ** A = new int * [ROWS];
    int ** C = new int * [ROWS];
    int ** B = new int * [ROWS];
    for (int i = 0; i < ROWS; i++) {
        A[i] = new int[COLS];
        B[i] = new int[COLS];
        C[i] = new int[COLS];
    }
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            A[i][j] = 1;
            B[i][j] = 12;
        }
    }

    auto _start = chrono::steady_clock::now();
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    auto _end = chrono::steady_clock::now();
    auto elapsed = _end - _start;

    cout << "rows = " << ROWS << "cols = " << COLS << endl;
    cout << "Time: ";
    cout << chrono::duration_cast< chrono::milliseconds > (elapsed).count() << "ms";
    cout << endl;

    // print res
    if (false) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                cout << C[i][j];
            }
            cout << endl;
        }    
    }

    return 0;
}
