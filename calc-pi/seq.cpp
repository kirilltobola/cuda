#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;

const int N = 1 << 20;
const double STEP = 1.0 / N;

double f(double x)
{
    return sqrt(1 - (x * x));
}

int main()
{
    auto _start = chrono::steady_clock::now();

    double sum_fi = 0;
    for (int i = 0; i < N; i++) {
        sum_fi += f(i * STEP);
    }
    double pi = 4 * (STEP * (0.5 + sum_fi));

    auto _end = chrono::steady_clock::now();
    auto elapsed = _end - _start;
    cout << "Time: ";
    cout << chrono::duration_cast< chrono::milliseconds > (elapsed).count() << "ms" << endl;
    cout << endl;

    cout << "Pi = " <<  pi;

    return 0;
}
