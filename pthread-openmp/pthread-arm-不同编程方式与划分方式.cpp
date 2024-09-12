#include<pthread.h>
#include<iostream>
#include<chrono>
#include<cmath>
#include <arm_neon.h>
#include <semaphore.h>
#include <stdio.h>
#include <sys/time.h>
#define ROW 3000
#define TASK 8
#define INTERVAL 10000
using namespace std;
using namespace chrono;
float a[ROW][ROW];
float rev[ROW][ROW];
typedef long long ll;
typedef struct {
	int k;
	int t_id;
}threadParam_t;

sem_t sem_leader;
sem_t sem_Divsion[32];
sem_t sem_Elimination[32];
pthread_barrier_t division;
pthread_barrier_t elemation;
int NUM_THREADS = 8;
int remain = ROW;
pthread_mutex_t remainLock;

void reverse()
{
	for (int i = 0; i < ROW; i++)
		for (int j = 0; j < ROW; j++)
			rev[j][i] = a[i][j];
}

void init()
{
	for (int i = 0; i < ROW; i++)
	{
		for (int j = 0; j < i; j++)
			a[i][j] = 0;
		for (int j = i; j < ROW; j++)
			a[i][j] = rand() / double(RAND_MAX) * 1000 + 1;
	}
	for (int k = 0; k < 1000; k++)
	{
		int row1 = rand() % ROW;
		int row2 = rand() % ROW;
		float mult = rand() & 1 ? 1 : -1;
		float mult2 = rand() & 1 ? 1 : -1;
		mult = mult2 * (rand() / double(RAND_MAX)) + mult;
		for (int j = 0; j < ROW; j++)
			a[row1][j] += mult * a[row2][j];
	}
	reverse();
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

void neon()
{
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&a[i][j]);
				float32x4_t mult2 = vld1q_f32(&a[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
	}
}

void* dynamicFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int j = k + 1; j < ROW; ++j)
		a[i][j] = a[i][j] - a[i][k] * a[k][j];
	a[i][k] = 0;
	pthread_exit(NULL);
}

void* dynamicFuncneon(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
	int j;
	for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
		a[i][j] = a[i][j] - a[i][k] * a[k][j];
	for (; j < ROW; j += 4)
	{
		float32x4_t sub1 = vld1q_f32(&a[i][j]);
		float32x4_t mult2 = vld1q_f32(&a[k][j]);
		mult2 = vmulq_f32(mult1, mult2);
		sub1 = vsubq_f32(sub1, mult2);
		vst1q_f32(&a[i][j], sub1);
	}
	a[i][k] = 0;
	pthread_exit(NULL);
}

void dynamicMain(void* (*threadFunc)(void*))
{
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		int worker_count = ROW - 1 - k;
		pthread_t* handles = new pthread_t[worker_count];
		threadParam_t* param = new threadParam_t[worker_count];
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* dynamicFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int i = k + t_id + 1; i < ROW; i += NUM_THREADS)
	{
		float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (; j < ROW; j += 4)
		{
			float32x4_t sub1 = vld1q_f32(&a[i][j]);
			float32x4_t mult2 = vld1q_f32(&a[k][j]);
			mult2 = vmulq_f32(mult1, mult2);
			sub1 = vsubq_f32(sub1, mult2);
			vst1q_f32(&a[i][j], sub1);
		}
		a[i][k] = 0;
	}
	pthread_exit(NULL);
}

void dynamicMain(void* (*threadFunc)(void*))
{
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		int worker_count = NUM_THREADS;
		pthread_t* handles = new pthread_t[worker_count];
		threadParam_t* param = new threadParam_t[worker_count];

		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}

		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);

		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* staticFunc(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		pthread_barrier_wait(&division);
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&a[i][j]);
				float32x4_t mult2 = vld1q_f32(&a[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void staticMain()
{
	pthread_barrier_init(&division, NULL, NUM_THREADS + 1);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS + 1);
	pthread_t* handles = new pthread_t[NUM_THREADS];
	for (long t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, staticFunc, (void*)t_id);

	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		pthread_barrier_wait(&division);
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void* staticFunc2(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		if (t_id == 0)
		{
			float32x4_t diver = vld1q_dup_f32(&a[k][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 4)
			{
				float32x4_t divee = vld1q_f32(&a[k][j]);
				divee = vdivq_f32(divee, diver);
				vst1q_f32(&a[k][j], divee);
			}
			a[k][k] = 1.0;
		}
		else sem_wait(&sem_Divsion[t_id - 1]);
		if (t_id == 0)
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_post(&sem_Divsion[t_id]);

		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&a[i][j]);
				float32x4_t mult2 = vld1q_f32(&a[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		if (t_id == 0)
		{
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_wait(&sem_leader);
			for (int t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
				sem_post(&sem_Elimination[t_id]);
		}
		else
		{
			sem_post(&sem_leader);
			sem_wait(&sem_Elimination[t_id - 1]);
		}
	}
	pthread_exit(NULL);
}

void staticMain2(void* (*threadFunc)(void*))
{
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < NUM_THREADS - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	pthread_t* handles = new pthread_t[NUM_THREADS];
	for (long t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);
	sem_destroy(&sem_leader);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		sem_destroy(&sem_Divsion[t_id]);
		sem_destroy(&sem_Elimination[t_id]);
	}
}

void* staticFunc_row(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		int count = (ROW - k - 1) / NUM_THREADS;
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		int endIt = k + 1 + count * (t_id + 1);
		for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < endIt; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		pthread_barrier_wait(&division);
		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&a[i][j]);
				float32x4_t mult2 = vld1q_f32(&a[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		pthread_barrier_wait(&elemation);
	}
	pthread_exit(NULL);
}

void staticMain_row(void* (*threadFunc)(void*))
{
	pthread_barrier_init(&division, NULL, NUM_THREADS);
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];
	long* param = new long[NUM_THREADS - 1];
	for (long t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;
		for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		pthread_barrier_wait(&division);
		for (int i = k + NUM_THREADS; i < ROW; i += NUM_THREADS)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			for (; j < ROW; j += 4)
			{
				float32x4_t sub1 = vld1q_f32(&a[i][j]);
				float32x4_t mult2 = vld1q_f32(&a[k][j]);
				mult2 = vmulq_f32(mult1, mult2);
				sub1 = vsubq_f32(sub1, mult2);
				vst1q_f32(&a[i][j], sub1);
			}
			a[i][k] = 0.0;
		}
		pthread_barrier_wait(&elemation);
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&division);
	pthread_barrier_destroy(&elemation);
}

void staticMain_col(void* (*threadFunc)(void*))
{
	pthread_barrier_init(&elemation, NULL, NUM_THREADS);
	pthread_t* handles = new pthread_t[NUM_THREADS - 1];
	long* param = new long[NUM_THREADS - 1];
	for (long t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;
		for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		for (int i = ROW - count; i < ROW; i++)
		{
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)
				a[j][i] = a[j][i] - a[j][k] * a[k][i];
			for (; j < ROW; j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_lane_f32(&a[j][i], subbee, 0);
				mult1 = vld1q_lane_f32(&a[j][k], mult1, 0);
				subbee = vld1q_lane_f32(&a[j + 1][i], subbee, 1);
				mult1 = vld1q_lane_f32(&a[j + 1][k], mult1, 1);
				subbee = vld1q_lane_f32(&a[j + 2][i], subbee, 2);
				mult1 = vld1q_lane_f32(&a[j + 2][k], mult1, 2);
				subbee = vld1q_lane_f32(&a[j + 3][i], subbee, 3);
				mult1 = vld1q_lane_f32(&a[j + 3][k], mult1, 3);
				mult2 = vld1q_dup_f32(&a[k][i]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_lane_f32(&a[j][i], subbee, 0);
				vst1q_lane_f32(&a[j + 1][i], subbee, 1);
				vst1q_lane_f32(&a[j + 2][i], subbee, 2);
				vst1q_lane_f32(&a[j + 3][i], subbee, 3);
			}
		}
		pthread_barrier_wait(&elemation);
		for (int j = k + NUM_THREADS; j < ROW; j += NUM_THREADS)
			a[j][k] = 0;
	}
	for (int t_id = 0; t_id < NUM_THREADS - 1; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&elemation);
}

void* staticFunc_col(void* param) {
	long t_id = (long)param;
	for (int k = 0; k < ROW; ++k)
	{
		int count = (ROW - k - 1) / NUM_THREADS;
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		int j;
		int endIt = k + 1 + count * (t_id + 1);
		for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < endIt; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		for (int i = k + 1 + t_id * count; i < k + count * (t_id + 1); i++)
		{
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)
				a[j][i] = a[j][i] - a[j][k] * a[k][i];
			for (; j < ROW; j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_lane_f32(&a[j][i], subbee, 0);
				mult1 = vld1q_lane_f32(&a[j][k], mult1, 0);
				subbee = vld1q_lane_f32(&a[j + 1][i], subbee, 1);
				mult1 = vld1q_lane_f32(&a[j + 1][k], mult1, 1);
				subbee = vld1q_lane_f32(&a[j + 2][i], subbee, 2);
				mult1 = vld1q_lane_f32(&a[j + 2][k], mult1, 2);
				subbee = vld1q_lane_f32(&a[j + 3][i], subbee, 3);
				mult1 = vld1q_lane_f32(&a[j + 3][k], mult1, 3);
				mult2 = vld1q_dup_f32(&a[k][i]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_lane_f32(&a[j][i], subbee, 0);
				vst1q_lane_f32(&a[j + 1][i], subbee, 1);
				vst1q_lane_f32(&a[j + 2][i], subbee, 2);
				vst1q_lane_f32(&a[j + 3][i], subbee, 3);
			}
		}
		pthread_barrier_wait(&elemation);
		for (int j = k + 1 + t_id; j < ROW; j += NUM_THREADS)
			a[j][k] = 0;
	}
	pthread_exit(NULL);
}

void timing(void(*func)())
{
	typedef std::chrono::high_resolution_clock Clock;
	double time = 0;
	init();
	auto t1 = Clock::now();
	func();
	auto t2 = Clock::now();
	time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6;
	cout << time << '\n';
}

void timing(void(*func)(void* (*threadFunc)(void*)), void* (*threadFunc)(void*))
{
	typedef std::chrono::high_resolution_clock Clock;
	double time = 0;

	init();
	auto t1 = Clock::now();
	func(threadFunc);
	auto t2 = Clock::now();
	time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6;
	cout << time << '\n';
}

int main()
{
	cout << "串行: ";
	timing(plain);
	cout << "neon: ";
	timing(neon);
	cout << "线程数: " << NUM_THREADS << endl;
	cout << "动态: ";
	timing(dynamicMain, dynamicFunc);
	cout << "静态信号量: ";
	timing(staticMain);
	cout << "静态信号量三重循环: ";
	timing(staticMain2, staticFunc2);
	cout << "静态barrier按行划分: ";
	timing(staticMain_row, staticFunc_row);
	cout << "静态barrier按列划分: ";
	timing(staticMain_col, staticFunc_col);
}