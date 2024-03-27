#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 5000; // matrix size

double b[N][N], sum[N];
double a[N];

void initb_nn(int n) 
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            b[i][j] = i + j;
}
void inita_n(int n) {
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }

}
int main()
{
    long long head, tail, freq; // timers
    inita_n(N);
    initb_nn(N);

    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    for (int i = 0; i < N; i++)
    {
        sum[i] = 0.0;   
    }
   

    for (int i = 0; i < N; i += 2) {
        for (int j = 0; j < N; j++)
        {
            sum[i] += b[j][i] * a[j];
            if (j + 1 < N) {
                sum[i + 1] = b[j + 1][i + 1] * a[j + 1];
            }
        }
    }


    for (int i = 0; i < N; i += 2) {
        
        for (int j = 0; j < N; j++) {
            
            sum[j] = b[i][j] * a[i];
            if (j + 1 < N) {
                sum[j + 1] = b[i + 1][j + 1] * a[i + 1];
            }
        }
    }

      // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    cout << "col:" << (tail - head)*1000.0 / freq
        << "ms" << endl;
}