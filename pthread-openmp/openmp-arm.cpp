#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <arm_neon.h>
#include <stdio.h>
#include<chrono>
#define ROW 2048
#define TASK 8
#define INTERVAL 10000
using namespace std;
using namespace chrono;
float a[ROW][ROW];
float rever[ROW][ROW];
typedef long long ll;
//静态线程数量
int NUM_THREADS = 4;

void rever()
{
	for (int i = 0; i < ROW; i++)
		for (int j = 0; j < ROW; j++)
			rever[j][i] = a[i][j];
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
	rever();
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
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < ROW; i++)
		{//消去
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			int j;
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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

void dynamic()//动态线程 
{
	int i, j, k;
	float32x4_t mult1, mult2, sub1;
	for (int k = 0; k < ROW; ++k)
	{
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)
			a[k][j] = a[k][j] / a[k][k];
		for (; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
#pragma omp parallel for num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
		for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void neonstatic()
{
	int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
	for (int k = 0; k < ROW; ++k)
	{
#pragma omp for
		for (j = k + 1; j < ROW; ++j)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
#pragma omp for
		for (int k = i + 1; k < ROW; k++) {
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
	}
}

void static_1()//静态线程 
{
	int i, j, k;
	float32x4_t mult1, mult2, sub1;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (int k = 0; k < ROW; ++k)
	{
#pragma omp single
		{
			float32x4_t diver = vld1q_dup_f32(&a[k][k]);
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
#pragma omp for
		for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static_2()//静态分除,不使用neon 
{
	int i, j, k;
	float32x4_t mult1, mult2, sub1;
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
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static_3()//静态分除,使用neon 
{
	int i, j, k, start;
	float32x4_t mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1), shared(start)
	for (k = 0; k < ROW; ++k) {
		float32x4_t diver = vld1q_dup_f32(&a[k][k]);
		//串行处理对齐 
#pragma omp single
		for (start = k + 1; start < ROW && ((ROW - start) & 3); ++start)
			a[k][start] = a[k][start] / a[k][k];
		//并行neon
#pragma omp for 
		for (j = start; j < ROW; j += 4)
		{
			float32x4_t divee = vld1q_f32(&a[k][j]);
			divee = vdivq_f32(divee, diver);
			vst1q_f32(&a[k][j], divee);
		}
		a[k][k] = 1.0;
		// 并行部分，使用行划分
#pragma omp for
		for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static_col()//列划分，cache优化前 
{
	int i, j, k;
	float32x4_t subbee, mult1, mult2;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, subbee, mult1, mult2)
	for (k = 0; k < ROW; ++k) {
#pragma omp for
		for (j = k + 1; j < ROW; ++j)
		{
			//除法 
			a[k][j] = a[k][j] / a[k][k];
			//消去 
			for (int i = k + 1; i < ROW; i++)//i为列
			{//逐列消去
				for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//处理串行部分
					a[j][i] = a[j][i] - a[j][k] * a[k][i];
				for (; j < ROW; j += 4)
				{
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
		}
#pragma omp single
		a[k][k] = 1.0;
#pragma omp for neon nowait
		for (i = k + 1; i < ROW; i++)
			a[i][k] = 0;
		// 不需要同步 
	}
}

void static_col_cache()//列划分，cache优化 
{
	int i, j, k;
	float32x4_t subbee, mult1, mult2;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, subbee, mult1, mult2)
	for (k = 0; k < ROW; ++k)
	{
#pragma omp for
		for (i = k + 1; i < ROW; i++)//i为列
		{//逐列消去
			rever[i][k] = rever[i][k] / rever[k][k];
			for (j = k + 1; j < ROW && ((ROW - j) & 3); j++)//处理串行部分
				rever[i][j] = rever[i][j] - rever[k][j] * rever[i][k];
			for (; j < ROW; j += 4)
			{
				float32x4_t subbee, mult1, mult2;
				subbee = vld1q_f32(&rever[i][j]);
				mult1 = vld1q_f32(&rever[k][j]);
				mult2 = vld1q_dup_f32(&rever[i][k]);
				mult1 = vmulq_f32(mult1, mult2);
				subbee = vsubq_f32(subbee, mult1);
				vst1q_f32(&rever[i][j], subbee);
			}
		}
#pragma omp single
		rever[k][k] = 1.0;
#pragma omp for neon nowait
		for (int j = k + 1; j < ROW; j++)
			rever[k][j] = 0;
		//处理已被消去的列
	}
}

void static_auto()//auto neon horizontal
{
	int i, j, k;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
	for (k = 0; k < ROW; ++k) {
		//并行化除法 
#pragma omp for
		for (j = k + 1; j < ROW; ++j)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		// 并行部分，使用行划分
#pragma omp for neon 
		for (i = k + 1; i < ROW; ++i)
		{
			for (int j = i + 1; j < ROW; j++) {
				a[k][j] = a[k][j] - a[i][j] * a[k][i];
			}
			a[k][i] = 0;
		}
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void static_auto_col()//auto neon col
{
	int i, j, k;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
	for (k = 0; k < ROW; ++k) {
		//并行化除法 
#pragma omp for neon
		for (j = k + 1; j < ROW; ++j)
		{
			a[k][j] = a[k][j] / a[k][k];
			for (i = k + 1; i < ROW; i++) {
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			}
		}
#pragma omp single
		a[k][k] = 1.0;
#pragma omp for neon nowait
		for (i = k + 1; i < ROW; i++)
			a[i][k] = 0;
		// 不需要同步 
	}
}

void static_10()//auto neon col cached
{
	int i, j, k;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k)
	for (k = 0; k < ROW; ++k)
	{
#pragma omp for
		for (i = k + 1; i < ROW; i++)//i为列
		{//逐列消去
			rever[i][k] = rever[i][k] / rever[k][k];
			for (j = k + 1; j < ROW; j++)//处理串行部分
				rever[i][j] = rever[i][j] - rever[k][j] * rever[i][k];
		}
#pragma omp single
		rever[k][k] = 1.0;
#pragma omp for neon nowait
		for (int j = k + 1; j < ROW; j++)
			rever[k][j] = 0;
		//处理已被消去的列
	}
}

void static_11()//静态任务划分
{
	int i, j, k;
	float32x4_t mult1, mult2, sub1;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
#pragma omp single
		{
			float32x4_t diver = vld1q_dup_f32(&a[k][k]);
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
#pragma omp for schedule(static, 3) 
		for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
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

void schedule_dynamic()//动态任务划分，块大小固定 
{
	int i, j, k;
	float32x4_t mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			float32x4_t diver = vld1q_dup_f32(&a[k][k]);
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
		// 并行部分，使用行划分
#pragma omp for schedule(dynamic, 3) 
		for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

void schedule_guided()//动态任务划分，guided 
{
	int i, j, k;
	float32x4_t mult1, mult2, sub1;
	// 在外循环之外创建线程，避免线程反复创建销毁，注意共享变量和私有变量的设置
	//删除下面一句即为动态线程 
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, mult1, mult2, sub1)
	for (k = 0; k < ROW; ++k) {
		// 串行部分，也可以尝试并行化
#pragma omp single
		{
			float32x4_t diver = vld1q_dup_f32(&a[k][k]);
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
		// 并行部分，使用行划分
#pragma omp for schedule(guided,3) 
		for (i = k + 1; i < ROW; ++i)
		{
			float32x4_t mult1 = vld1q_dup_f32(&a[i][k]);
			for (j = k + 1; j < ROW && ((ROW - j) & 3); ++j)//串行处理对齐
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
	cout << "串行：";
	timing(plain);
	cout << "neon：";
	timing(neon);
	for (NUM_THREADS = 8; NUM_THREADS <= 8; NUM_THREADS++)
	{
		// cout<<"using "<<NUM_THREADS<<" threads"<<endl; 
		// cout<<"动态："; 
		// timing(dynamic);
		// cout<<"静态1："; 
		//timing(static_1);
		cout << "静态2：";
		timing(static_2);
		cout << "静态3：";
		timing(static_3);
		// cout<<"静态按列："; 
		// timing(static_col);
		// cout<<"静态按列cached："; 
		// timing(static_col_cache);
		cout << "自动向量化：";
		timing(static_auto);
		cout << "自动向量化按列cache：";
		timing(static_10);
		// cout<<"自动向量化按列：";
		// timing(static_auto_col); 
		cout << "循环划分dynamic：";
		timing(schedule_dynamic);
		cout << "循环划分guided：";
		timing(schedule_guided);
	}
}