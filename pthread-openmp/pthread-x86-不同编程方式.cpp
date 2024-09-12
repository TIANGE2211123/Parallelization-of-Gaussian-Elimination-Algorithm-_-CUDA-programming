#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX?1?7?1?7AVX2
#include<pthread.h>
#include<iostream>
#include<cmath>
#include <semaphore.h>
#include <stdio.h>
#include <windows.h>
#define ROW 1024
#define TASK 10
#define INTERVAL 10000
using namespace std;
float a[ROW][ROW];
typedef long long ll;
typedef struct {
	int k;
	int t_id;
}threadParam_t;

sem_t sem_main;
sem_t sem_workerstart;
sem_t sem_workerend;
//信号量定义，用于静态线程
sem_t sem_leader;
sem_t sem_Divsion[32];
sem_t sem_Elimination[32];
pthread_barrier_t division;
pthread_barrier_t elemation;
int NUM_THREADS = 4;
//动态线程分配：待分配行数
int remain = ROW;
pthread_mutex_t remainLock;

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

void sse()
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

void* newDynamicFuncsse(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	int i = k + t_id + 1;
	for (int i = k + t_id + 1; i < ROW; i += NUM_THREADS)
	{
		__m128 mult1 = _mm_load_ss(&a[i][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
			a[i][j] = a[i][j] - a[i][k] * a[k][j];
		for (; j < ROW; j += 4)
		{
			__m128 sub1 = _mm_loadu_ps(&a[i][j]);
			__m128 mult2 = _mm_loadu_ps(&a[k][j]);
			mult2 = _mm_mul_ps(mult1, mult2);
			sub1 = _mm_sub_ps(sub1, mult2);
			_mm_storeu_ps(&a[i][j], sub1);
		}
		a[i][k] = 0;
	}
	pthread_exit(NULL);
}

void newDynamicMain(void* (*threadFunc)(void*))
{
	for (int k = 0; k < ROW; ++k)
	{//主线程做除法操作
		__m128 diver = _mm_load_ss(&a[k][k]);
		int j;
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			__m128 divee = _mm_loadu_ps(&a[k][j]);
			divee = _mm_div_ps(divee, diver);
			_mm_storeu_ps(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		//创建工作线程，进行消去操作
		int worker_count = NUM_THREADS; //工作线程数量
		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构
		//分配任务
		for (long t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (long t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (long t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);
	}
}

void* staticFunc(void* param) {
	long t_id = (long long)param;
	for (int k = 0; k < ROW; ++k)
	{
		//sem_wait(&sem_workerstart); 
		pthread_barrier_wait(&division);

		for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
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
		pthread_barrier_wait(&elemation);
		//sem_post(&sem_main);    	
//sem_wait(&sem_workerend); 	}
		pthread_exit(NULL);
	}

	void staticMain()
	{
		//初始化barrier
		pthread_barrier_init(&division, NULL, NUM_THREADS + 1);
		pthread_barrier_init(&elemation, NULL, NUM_THREADS + 1);
		pthread_t* handles = new pthread_t[NUM_THREADS];
		for (long t_id = 0; t_id < NUM_THREADS; t_id++)
			pthread_create(&handles[t_id], NULL, staticFunc, (void*)t_id);

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
			pthread_barrier_wait(&division);
			pthread_barrier_wait(&elemation);
		}
		for (long t_id = 0; t_id < NUM_THREADS; t_id++)
			pthread_join(handles[t_id], NULL);
		pthread_barrier_destroy(&division);
		pthread_barrier_destroy(&elemation);
	}

	void* staticFuncOpt(void* param) {
		long t_id = (long long)param;
		for (int k = 0; k < ROW; ++k)
		{

			if (t_id == 0)
			{
				__m128 diver = _mm_set1_ps(a[k][k]);
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
			}
			else sem_wait(&sem_Divsion[t_id - 1]);
			if (t_id == 0)
				for (long t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
					sem_post(&sem_Divsion[t_id]);

			for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
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

			if (t_id == 0)
			{
				for (long t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
					sem_wait(&sem_leader);
				for (long t_id = 0; t_id < NUM_THREADS - 1; ++t_id)
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

	void staticOptMain(void* (*threadFunc)(void*))
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
		for (long t_id = 0; t_id < NUM_THREADS; t_id++)
			pthread_join(handles[t_id], NULL);
		sem_destroy(&sem_leader);
		for (long t_id = 0; t_id < NUM_THREADS; t_id++)
		{
			sem_destroy(&sem_Divsion[t_id]);
			sem_destroy(&sem_Elimination[t_id]);
		}
	}

	void* staticFuncOptNew(void* param) {
		long t_id = (long long)param;
		for (int k = 0; k < ROW; ++k)
		{

			int count = (ROW - k - 1) / NUM_THREADS;
			__m128 diver = _mm_load_ps1(&a[k][k]);
			int j;

			int endIt = k + 1 + count * (t_id + 1);//?1?7?1?7?1?7?1?7?0?6?1?7?1?7
			for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)
				a[k][j] = a[k][j] / a[k][k];
			for (; j < endIt; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			pthread_barrier_wait(&division);

			for (int i = k + 1 + t_id; i < ROW; i += NUM_THREADS)
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

			pthread_barrier_wait(&elemation);
		}
		pthread_exit(NULL);
	}

	void staticNewOptMain(void* (*threadFunc)(void*))
	{
		//初始化barrier
		pthread_barrier_init(&division, NULL, NUM_THREADS);
		pthread_barrier_init(&elemation, NULL, NUM_THREADS);
		//创建线程
		pthread_t* handles = new pthread_t[NUM_THREADS - 1];// 创建对应的 Handle
		long* param = new long[NUM_THREADS - 1];// 创建对应的线程数据结构
		for (long t_id = 0; t_id < NUM_THREADS - 1; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
		//主函数看作第NUM_THREADS-1号线程
		for (int k = 0; k < ROW; ++k)
		{
			__m128 diver = _mm_load_ss(&a[k][k]);
			int j;
			int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
			//主线程处理ROW-count~ROW-1
			for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			a[k][k] = 1.0;
			//主线程睡眠（等待所有的工作线程完成此轮消去任务）
			pthread_barrier_wait(&division);
			for (int i = k + NUM_THREADS; i < ROW; i += NUM_THREADS)
			{//消去
				__m128 mult1 = _mm_load_ss(&a[i][k]);
				int j;
				for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
			// 所有线程一起进入下一轮
			pthread_barrier_wait(&elemation);
		}
		for (long t_id = 0; t_id < NUM_THREADS - 1; t_id++)
			pthread_join(handles[t_id], NULL);
		pthread_barrier_destroy(&division);
		pthread_barrier_destroy(&elemation);
	}

	void DynamicDivMain(void* (*threadFunc)(void*))
	{
		//初始化锁
		pthread_mutex_init(&remainLock, NULL);
		//初始化barrier
		pthread_barrier_init(&division, NULL, NUM_THREADS);
		pthread_barrier_init(&elemation, NULL, NUM_THREADS);
		//创建线程
		pthread_t* handles = new pthread_t[NUM_THREADS - 1];// 创建对应的 Handle
		long* param = new long[NUM_THREADS - 1];// 创建对应的线程数据结构
		for (long t_id = 0; t_id < NUM_THREADS - 1; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)t_id);
		//主函数看作第NUM_THREADS-1号线程
		for (int k = 0; k < ROW; ++k)
		{
			__m128 diver = _mm_load_ss(&a[k][k]);
			int j;
			int count = (ROW - k - 1) / NUM_THREADS + (ROW - k - 1) % NUM_THREADS;//主线程要处理的数量
			//主线程处理ROW-count~ROW-1
			for (j = ROW - count; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
				a[k][j] = a[k][j] / a[k][k];
			for (; j < ROW; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			a[k][k] = 1.0;
			//发出任务
			remain = k + 1;
			//等待子线程就位
			pthread_barrier_wait(&division);
			while (true)
			{
				int i;//行
				//处理i`i+TASK - 1
				//领取任务
				pthread_mutex_lock(&remainLock);
				if (remain >= ROW)
				{
					pthread_mutex_unlock(&remainLock);
					break;
				}
				i = remain;
				remain += TASK;
				pthread_mutex_unlock(&remainLock);
				int end = min(ROW, i + TASK);
				for (; i < end; i++)
				{
					//消去
					__m128 mult1 = _mm_load_ss(&a[i][k]);
					int j;//列
					for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
			// 所有线程一起进入下一轮
			pthread_barrier_wait(&elemation);
		}
		for (long t_id = 0; t_id < NUM_THREADS - 1; t_id++)
			pthread_join(handles[t_id], NULL);
		pthread_barrier_destroy(&division);
		pthread_barrier_destroy(&elemation);
		pthread_mutex_destroy(&remainLock);
	}

	void* DynamicDivFunc(void* param) {
		long t_id = (long long)param;
		for (int k = 0; k < ROW; ++k)
		{
			//除法
			int count = (ROW - k - 1) / NUM_THREADS;
			__m128 diver = _mm_load_ss(&a[k][k]);
			int j;
			//子线程处理k+1+count*t_id~k+count*(t_id+1)
			int endIt = k + 1 + count * (t_id + 1);//向量末端
			for (j = k + 1 + count * t_id; j < endIt && ((endIt - j) & 3); ++j)//串行处理对齐
				a[k][j] = a[k][j] / a[k][k];
			for (; j < endIt; j += 4)
			{
				__m128 divee = _mm_loadu_ps(&a[k][j]);
				divee = _mm_div_ps(divee, diver);
				_mm_storeu_ps(&a[k][j], divee);
			}
			pthread_barrier_wait(&division);
			//循环划分任务（同学们可以尝试多种任务划分方式）
			while (true)
			{
				int i;
				pthread_mutex_lock(&remainLock);
				if (remain >= ROW)
				{
					pthread_mutex_unlock(&remainLock);
					break;
				}
				i = remain;
				remain += TASK;
				pthread_mutex_unlock(&remainLock);
				int end = min(ROW, i + TASK);
				for (; i < end; i++)
				{//消去
					__m128 mult1 = _mm_load_ss(&a[i][k]);
					int j;
					for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
			// 所有线程一起进入下一轮
			pthread_barrier_wait(&elemation);
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
		cout << "sse: ";
		timing(sse);
		for (NUM_THREADS = 8; NUM_THREADS <= 8; NUM_THREADS++)
		{
			cout << "线程数: " << NUM_THREADS << endl;
			cout << "动态: ";
			timing(newDynamicMain, newDynamicFuncsse);
			cout << "静态信号量: ";
			timing(staticMain);
			cout << "静态信号量三循环: ";
			timing(staticOptMain, staticFuncOpt);
			cout << "静态barrier: ";
			timing(staticNewOptMain, staticFuncOptNew);
		}
	}