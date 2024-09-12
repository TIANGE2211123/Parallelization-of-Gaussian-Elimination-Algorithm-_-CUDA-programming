#include <iostream>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX
#include <omp.h>
#include<chrono>
#include <stdio.h>
//#include <windows.h>
using namespace std;
using namespace chrono;
typedef long long ll;

#define ROW 1024 
#define chunk_size 3
#define TASK 10
#define INTERVAL 10000
float a[ROW][ROW];
int NUM_THREADS = 8;

//动态线程分配：待分配行数
int remain = ROW;

void init()
{
	for (int i = 0; i < ROW; i++)
	{
		for (int j = 0; j < i; j++)
			a[i][j] = 0;
		for (int j = i; j < ROW; j++)
			a[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0; k < 8000; k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0; j < ROW; j++)
			a[row1][j] += mult * a[row2][j];
	}
}

void plain() {
	for (int i = 0; i < ROW - 1; i++) {
		for (int j = i + 1; j < ROW; j++) {
			a[i][j] = a[i][j] / a[i][i];
		}
		a[i][i] = 1;
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
	}
}

void SIMD()
{
	for (int k = 0; k < ROW; ++k)
	{
		__m128 diver = _mm_load_ps1(&a[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			__m128 divee = _mm_loadu_ps(&a[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i += 1)
		{
			__m128 mult1 = _mm_load_ps1(&a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				__m128 sub1 = _mm_loadu_ps(&a[i][j]);
				__m128 mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
	}
}

void dynamic()//动态限制
{
	int i, j, k;
	for (k = 0; k < ROW; ++k) {
		for (int j = i + 1; j < ROW; j++) {
			a[i][j] = a[i][j] / a[i][i];
		}
		a[i][i] = 1;
#pragma omp parallel for num_threads(NUM_THREADS) 
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
	}
}


void dynamic_sse()//动态限制
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
		__m128 diver = _mm_load_ps1(&a[k][k]);
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			__m128 divee = _mm_loadu_ps(&a[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		// 并行部分，使用行划分
#pragma omp parallel for num_threads(NUM_THREADS) 
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				sub1 = _mm_loadu_ps(&a[i][j]);
				mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}



void dynamic_avx()//动态限制
{
	int i, j, k;
	__m256 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
		__m256 diver = _mm256_set1_ps(a[k][k]);
		for (j = k + 1; j < ROW && ((ROW - j) & 7); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 8)
		{
			__m256 divee = _mm256_loadu_ps(&a[k][j]);
			divee = _mm256_div_ps(divee, diver);
			_mm256_storeu_ps(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		// 并行部分，使用行划分
#pragma omp parallel for num_threads(NUM_THREADS) 
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm256_set1_ps(a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 7); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 8)
			{
				__m256 sub1 = _mm256_loadu_ps(&a[i][j]);
				__m256 mult2 = _mm256_loadu_ps(&a[k][j]);
				mult2 = _mm256_mul_ps(mult1, mult2);
				sub1 = _mm256_sub_ps(sub1, mult2);
				_mm256_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static1()//静态朴素 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			for (int j = i + 1; j < ROW; j++) {
				a[i][j] = a[i][j] / a[i][i];
			}
			a[i][i] = 1;
		}
		// 并行部分，使用行划分
#pragma omp for
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static_sse()//静态朴素 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			__m128 diver = _mm_load_ps1(&a[k][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			a[k][k] = 1.0;
		}
		// 并行部分，使用行划分
#pragma omp for
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				sub1 = _mm_loadu_ps(&a[i][j]);
				mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static_avx()//静态朴素 
{
	int i, j, k;
	__m256 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			__m256 diver = _mm256_set1_ps(a[k][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 7); ++j)
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 8)
			{
				__m256 divee = _mm256_loadu_ps(&a[k][j]);
				divee = _mm256_div_ps(divee, diver);
				_mm256_storeu_ps(&a[k][j], divee);
			}
			a[k][k] = 1.0;
		}
		// 并行部分，使用行划分
#pragma omp for
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm256_set1_ps(a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				__m256 sub1 = _mm256_loadu_ps(&a[i][j]);
				__m256 mult2 = _mm256_loadu_ps(&a[k][j]);
				mult2 = _mm256_mul_ps(mult1, mult2);
				sub1 = _mm256_sub_ps(sub1, mult2);
				_mm256_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}
void avx()
{
	for (int k = 0; k < ROW; ++k)
	{
		__m256 diver = _mm256_set1_ps(a[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 7); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 8)
		{
			__m256 divee = _mm256_loadu_ps(&a[k][j]);
			divee = _mm256_div_ps(divee, diver);
			_mm256_storeu_ps(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i += 1)
		{
			__m256 mult1 = _mm256_set1_ps(a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 7); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 8)
			{
				__m256 sub1 = _mm256_loadu_ps(&a[i][j]);
				__m256 mult2 = _mm256_loadu_ps(&a[k][j]);
				mult2 = _mm256_mul_ps(mult1, mult2);
				sub1 = _mm256_sub_ps(sub1, mult2);
				_mm256_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
	}
}

void static2()//静态分除,不使用SIMD 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 并行除法
#pragma omp for
		for (j = k + 1; j < ROW; ++j)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		// 并行部分，使用行划分
#pragma omp for
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				sub1 = _mm_loadu_ps(&a[i][j]);
				mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static3()//静态分除,使用SIMD 
{
	int i, j, k, start;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1), shared(start)
	for (k = 0; k < ROW; ++k) {
		__m128 diver = _mm_load_ps1(&a[k][k]);
		//串行处理对齐 
#pragma omp single
		for (start = k + 1; start < ROW && ((ROW - start) & 3); ++start)
			a[k][start] = a[k][start] / a[k][k];
		//并行SIMD
#pragma omp for 
		for (j = start; j < ROW; j += 4)
		{
			__m128 divee = _mm_loadu_ps(&a[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		// 并行部分，使用行划分
#pragma omp for
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				sub1 = _mm_loadu_ps(&a[i][j]);
				mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void schedule_static()//动态任务划分，块大小固定 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			__m128 diver = _mm_load_ps1(&a[k][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			a[k][k] = 1.0;
		}
		// 并行部分，使用行划分
#pragma omp for schedule(static, chunk_size) 
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				sub1 = _mm_loadu_ps(&a[i][j]);
				mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void schedule_dynamic()//动态任务划分，块大小固定 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			__m128 diver = _mm_load_ps1(&a[k][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			a[k][k] = 1.0;
		}
		// 并行部分，使用行划分
#pragma omp for schedule(dynamic, chunk_size) 
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				sub1 = _mm_loadu_ps(&a[i][j]);
				mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void schedule_guided()//动态任务划分，guided 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			__m128 diver = _mm_load_ps1(&a[k][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			a[k][k] = 1.0;
		}
		// 并行部分，使用行划分
#pragma omp for schedule(guided,chunk_size) 
		for (i = k + 1; i < ROW; ++i)
		{
			mult1 = _mm_load_ps1(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				sub1 = _mm_loadu_ps(&a[i][j]);
				mult2 = _mm_loadu_ps(&a[k][j]);
				mult2 = _mm_mul_ps(mult1, mult2);
				sub1 = _mm_sub_ps(sub1, mult2);
				_mm_storeu_ps(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void schedule_static1()
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			for (int j = i + 1; j < ROW; j++) {
				a[i][j] = a[i][j] / a[i][i];
			}
		}
		// 并行部分，使用行划分
#pragma omp for schedule(static, chunk_size) 
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void schedule_dynamic1()//动态划分{
int i, j, k;
__m128 mult1, mult2, sub1;
// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
for (k = 0; k < ROW; ++k) {
	// 串行部分，也可以尝试并行化
#pragma omp single
	{for (int j = i + 1; j < ROW; j++) {
		a[i][j] = a[i][j] / a[i][i];
	}
	}
	// 并行部分，使用行划分
#pragma omp for schedule(dynamic, chunk_size) 
	for (int k = i + 1; k < ROW; k++) {
		for (int j = i + 1; j < ROW; j++) {
			a[k][j] = a[k][j] - a[i][j] * a[k][i];
		}
		a[k][i] = 0;
	}
	// 离开for循环时，各个线程默认同步，进入下一行的处理
}
}

void schedule_guided1()//guided 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			for (int j = i + 1; j < ROW; j++) {
				a[i][j] = a[i][j] / a[i][i];
			}
		}
		// 并行部分，使用行划分
#pragma omp for schedule(guided,chunk_size) 
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}



void timing(void(*func)())
{
	typedef std::chrono::high_resolution_clock Clock;
	double time = 0;
	init();
	auto t1 = Clock::now();
	func();
	auto t2 = Clock::now();
	time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6;
	cout << time << '\n';
}

int main()
{
	cout << "N=" << ROW << endl;
	cout << "串行：";
	timing(plain);
	//cout<<"SIMD："; 
	//timing(SIMD);
	for (NUM_THREADS = 8; NUM_THREADS <= 8; NUM_THREADS += 2)
	{
		cout << "using " << NUM_THREADS << " threads" << endl;
		cout << "动态：";
		timing(dynamic);
		cout << "动态sse：";
		timing(dynamic_sse);
		cout << "动态avx：";
		timing(dynamic_avx);
		cout << "静态1：";
		timing(static1);
		cout << "静态sse：";
		timing(static_sse);
		cout << "静态avx：";
		timing(static_avx);
		cout << "静态2：";
		timing(static2);
		cout << "静态3：";
		timing(static3);
		cout << "static：";
		timing(schedule_static1);
		cout << "dynamic：";
		timing(schedule_dynamic1);
		cout << "guided：";
		timing(schedule_guided1);
	}
}