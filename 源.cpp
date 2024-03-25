
#include <iostream>
#include <windows.h>
#include <stdlib.h>

 using namespace std;

 const int N = 1000000; 
 double a[N];
 int sum0, sum1, sum2 = 0;


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
     
     Sleep(1000);
     // similar to CLOCKS_PER_SEC
         QueryPerformanceFrequency((LARGE_INTEGER*) & freq);
     // start time
         QueryPerformanceCounter((LARGE_INTEGER*) & head);

         inita_n(N);
         int t = 50;
     
         
         //循环延时
             for (int i = 0; i < N; i++)   //平凡算法
             {
                 sum0 += a[i];
             }
         
         

        
         //for(int i=0;i<N;i+=2)
         //{
         //    sum1+=a[i];          //优化算法
         //    sum2+=a[i+1];
         //}
         //
         //sum0=sum1+sum2;
         

     
     // end time
         QueryPerformanceCounter((LARGE_INTEGER *) & tail);

     cout << "col:" << (tail - head) * 1000.0 / freq
         << "ms" << endl;
     }




//#include <iostream>
//#include <windows.h>
//#include <stdlib.h>
//
//using namespace std;
//
//const int N = 5000; // matrix size
//
//double b[N][N], sum[N];
//double a[N];
//
//void initb_nn(int n) // generate a N∗N matrix
//{
//    for (int i = 0; i < N; i++)
//        for (int j = 0; j < N; j++)
//            b[i][j] = i + j;
//}
//void inita_n(int n) {
//    for (int i = 0; i < N; i++)
//    {
//        a[i] = i;
//    }
//
//}
//int main()
//{
//    long long head, tail, freq; // timers
//    inita_n(N);
//    initb_nn(N);
//
//    // similar to CLOCKS_PER_SEC
//    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
//    // start time
//    QueryPerformanceCounter((LARGE_INTEGER*)&head);
//
//    for (int i = 0; i < N; i++)
//    {
//        sum[i] = 0.0;
//    
//    }
//    //{
//    // for (int i = 0; i < N; i++) {   
//    //   for (int j = 0; j < N; j++)  //平凡算法，逐列访问
//    //   { 
//    //     sum[i] += b[j][i] * a[j];
//    //    }
//    //   }
//    //}
//    
//    
//    
//    for (int i = 0; i < N; i++)
//    {
//        for (int j = 0; j < N; j++)         //cache优化,逐行访问
//        {
//            sum[j] += b[i][j] * a[i];
//        }
//    }
//
//
//
//      // end time
//    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
//
//    cout << "col:" << (tail - head)*1000.0 / freq
//        << "ms" << endl;
//}
