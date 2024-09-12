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

//��̬�̷߳��䣺����������
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

void dynamic()//��̬����
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


void dynamic_sse()//��̬����
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
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
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}



void dynamic_avx()//��̬����
{
	int i, j, k;
	__m256 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
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
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void static1()//��̬���� 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
#pragma omp single
		{
			for (int j = i + 1; j < ROW; j++) {
				a[i][j] = a[i][j] / a[i][i];
			}
			a[i][i] = 1;
		}
		// ���в��֣�ʹ���л���
#pragma omp for
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void static_sse()//��̬���� 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
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
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void static_avx()//��̬���� 
{
	int i, j, k;
	__m256 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
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
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
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

void static2()//��̬�ֳ�,��ʹ��SIMD 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���г���
#pragma omp for
		for (j = k + 1; j < ROW; ++j)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void static3()//��̬�ֳ�,ʹ��SIMD 
{
	int i, j, k, start;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1), shared(start)
	for (k = 0; k < ROW; ++k) {
		__m128 diver = _mm_load_ps1(&a[k][k]);
		//���д������ 
#pragma omp single
		for (start = k + 1; start < ROW && ((ROW - start) & 3); ++start)
			a[k][start] = a[k][start] / a[k][k];
		//����SIMD
#pragma omp for 
		for (j = start; j < ROW; j += 4)
		{
			__m128 divee = _mm_loadu_ps(&a[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void schedule_static()//��̬���񻮷֣����С�̶� 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
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
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void schedule_dynamic()//��̬���񻮷֣����С�̶� 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
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
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void schedule_guided()//��̬���񻮷֣�guided 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
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
		// ���в��֣�ʹ���л���
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
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void schedule_static1()
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
#pragma omp single
		{
			for (int j = i + 1; j < ROW; j++) {
				a[i][j] = a[i][j] / a[i][i];
			}
		}
		// ���в��֣�ʹ���л���
#pragma omp for schedule(static, chunk_size) 
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
	}
}

void schedule_dynamic1()//��̬����{
int i, j, k;
__m128 mult1, mult2, sub1;
// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
for (k = 0; k < ROW; ++k) {
	// ���в��֣�Ҳ���Գ��Բ��л�
#pragma omp single
	{for (int j = i + 1; j < ROW; j++) {
		a[i][j] = a[i][j] / a[i][i];
	}
	}
	// ���в��֣�ʹ���л���
#pragma omp for schedule(dynamic, chunk_size) 
	for (int k = i + 1; k < ROW; k++) {
		for (int j = i + 1; j < ROW; j++) {
			a[k][j] = a[k][j] - a[i][j] * a[k][i];
		}
		a[k][i] = 0;
	}
	// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
}
}

void schedule_guided1()//guided 
{
	int i, j, k;
	__m128 mult1, mult2, sub1;
	// ����ѭ��֮�ⴴ���̣߳������̷߳����������٣�ע�⹲�������˽�б���������
	//ɾ������һ�伴Ϊ��̬�߳� 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// ���в��֣�Ҳ���Գ��Բ��л�
#pragma omp single
		{
			for (int j = i + 1; j < ROW; j++) {
				a[i][j] = a[i][j] / a[i][i];
			}
		}
		// ���в��֣�ʹ���л���
#pragma omp for schedule(guided,chunk_size) 
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
		// �뿪forѭ��ʱ�������߳�Ĭ��ͬ����������һ�еĴ���
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
	cout << "���У�";
	timing(plain);
	//cout<<"SIMD��"; 
	//timing(SIMD);
	for (NUM_THREADS = 8; NUM_THREADS <= 8; NUM_THREADS += 2)
	{
		cout << "using " << NUM_THREADS << " threads" << endl;
		cout << "��̬��";
		timing(dynamic);
		cout << "��̬sse��";
		timing(dynamic_sse);
		cout << "��̬avx��";
		timing(dynamic_avx);
		cout << "��̬1��";
		timing(static1);
		cout << "��̬sse��";
		timing(static_sse);
		cout << "��̬avx��";
		timing(static_avx);
		cout << "��̬2��";
		timing(static2);
		cout << "��̬3��";
		timing(static3);
		cout << "static��";
		timing(schedule_static1);
		cout << "dynamic��";
		timing(schedule_dynamic1);
		cout << "guided��";
		timing(schedule_guided1);
	}
}