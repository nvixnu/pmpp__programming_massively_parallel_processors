/*
 ============================================================================
 Name        : main.cu
 Author      : Nvixnu
 Version     :
 Copyright   : 
 Description : Programming massively Parallel Processors with CUDA - 3ed.
 ============================================================================
 */

#include <stdio.h>
#include "nvixnu__cuda_devices_props.h"
#include "chapter_02/ch2__config.h"
#include "chapter_03/ch3__config.h"
#include "chapter_04/ch4__config.h"
#include "chapter_05/ch5__config.h"
#include "chapter_07/ch7__config.h"
#include "chapter_08/ch8__config.h"
#include "chapter_09/ch9__config.h"
#include "chapter_10/ch10__config.h"
#include "chapter_11/ch11__config.h"
#include "chapter_12/ch12__config.h"
#include "utils.h"


static inline void chapter_2_menu(){
	printf("Chapter 02\n");
	printf("Running [vec_add] on Device with 256 threads per block...:\n");
	ch2__vec_add(Device, {.block_dim = {256,1,1}});

	printf("\nRunning [vec_add] on Device with 1024 threads per block...:\n");
	ch2__vec_add(Device, {.block_dim = {1024,1,1}});

	printf("\nRunning [vec_add] on Host...\n");
	ch2__vec_add(Host, {});
}


static inline void chapter_3_menu(){
	int option = -1;
	while(option != 0){
		printf("\nCHAPTER 03:\n");
		switch(option){
		case 1:
			printf("Running [color_to_grayscale] on Device with 256 threads per block...:\n");
			ch3__color_to_grayscale(Device, {.block_dim = {16,16,1}});

			printf("\nRunning [color_to_grayscale] on Device with 1024 threads per block...:\n");
			ch3__color_to_grayscale(Device, {.block_dim = {32,32,1}});

			printf("\nRunning [color_to_grayscale] on Host...\n");
			ch3__color_to_grayscale(Host, {});
			option = -1;
			break;
		case 2:
			printf("Running [blur] on Device with 256 threads per block...:\n");
			ch3__blur(Device, {.block_dim = {16,16,1}});

			printf("\nRunning [blur] on Device with 1024 threads per block...:\n");
			ch3__blur(Device, {.block_dim = {32,32,1}});

			printf("\nRunning [blur] on Host...\n");
			ch3__blur(Host, {});
			option = -1;
			break;
		default:
			printf("\t\t[1] - Color to grayscale\n");
			printf("\t\t[2] - Blur\n");
			printf("\nPress the number of the algorithm or zero to go back.\n");
			scanf("%d", &option);
			setbuf(stdin, NULL);
		}
	}
}

static inline void chapter_4_menu(){
	printf("Chapter 04\n");
	printf("Running [matrix_mul] on Device with 256 threads per block...:\n");
	ch4__matrix_mul(Device, {.block_dim = {16,16,1}, .kernel_version = CH4__MATRIX_MUL_KERNEL_NAIVE});

	printf("\nRunning [matrix_mul] on Device with 1024 threads per block...:\n");
	ch4__matrix_mul(Device, {.block_dim = {32,32,1}, .kernel_version = CH4__MATRIX_MUL_KERNEL_NAIVE});

	printf("\nRunning [matrix_mul_tiled] on Device with 256 threads per block...:\n");
	ch4__matrix_mul(Device, {.block_dim = {16,16,1}, .kernel_version = CH4__MATRIX_MUL_KERNEL_TILED});

	printf("\nRunning [matrix_mul_tiled] on Device with 1024 threads per block...:\n");
	ch4__matrix_mul(Device, {.block_dim = {32,32,1}, .kernel_version = CH4__MATRIX_MUL_KERNEL_TILED});

	printf("\nRunning [matrix_mul] on Host...\n");
	ch4__matrix_mul(Host, {});
}

static inline void chapter_5_menu(){
	printf("Chapter 05\n");
	printf("Running [ch5__sum_reduction] on Device with 256 threads per block...:\n");
	ch5__sum_reduction(Device, {.block_dim = {256,1,1}});

	printf("\nRunning [ch5__sum_reduction] on Device with 1024 threads per block...:\n");
	ch5__sum_reduction(Device, {.block_dim = {1024,1,1}});

	printf("\nRunning [ch5__sum_reduction] on Host...\n");
	ch5__sum_reduction(Host, {});
}

static inline void chapter_7_menu(){
	int option = -1;
	while(option != 0){
		printf("\nCHAPTER 07:\n");
		switch(option){
		case 1:
			printf("Running [ch7__1d_convolution] on Device with 256 threads per block...:\n");
			ch7__1d_convolution(Device, {.block_dim = {256,1,1}});

			printf("\nRunning [ch7__1d_convolution] on Device with 1024 threads per block...:\n");
			ch7__1d_convolution(Device, {.block_dim = {1024,1,1}});

			printf("\nRunning [ch7__1d_convolution] on Host...\n");
			ch7__1d_convolution(Host, {});
			option = -1;
			break;
		case 2:
			printf("Running [ch7__2d_convolution] on Device with 256 threads per block...:\n");
			ch7__2d_convolution(Device, {.block_dim = {16,16,1}});

			printf("\nRunning [ch7__2d_convolution] on Device with 1024 threads per block...:\n");
			ch7__2d_convolution(Device, {.block_dim = {32,32,1}});

			printf("\nRunning [ch7__2d_convolution] on Host...\n");
			ch7__2d_convolution(Host, {});
			option = -1;
			break;
		default:
			printf("\t\t[1] - 1D convolution\n");
			printf("\t\t[2] - 2D convolution\n");
			printf("\nPress the number of the algorithm or zero to go back.\n");
			scanf("%d", &option);
			setbuf(stdin, NULL);
		}
	}
}

static inline void chapter_8_menu(){
	int option = -1;
	//Gets the max length of shared memory to use as SECTION_SIZE of the 3-phase algorithm
	cudaDeviceProp device_props =  nvixnu__get_cuda_device_props(0);
	const int memory_bound_section_size = device_props.sharedMemPerBlock;
	const int memory_bound_section_length = memory_bound_section_size/sizeof(double);
	const int thread_bound_section_length = device_props.maxThreadsDim[0];

	while(option != 0){
		printf("\nCHAPTER 08:\n");
		printf("CH8__ARRAY_LENGTH: %d\n", CH8__ARRAY_LENGTH);
		switch(option){
		case 1:

			printf("\nRunning [ch8__partial_prefix_sum Kogge-Stone] on Device with %d threads per block...:\n", thread_bound_section_length);
			ch8__partial_prefix_sum(Device, {
					.block_dim = {thread_bound_section_length,1,1},
					.kernel_version = CH8__PREFIX_SUM_KOGGE_STONE
			}, 0);

			printf("\nRunning [ch8__partial_prefix_sum Brent-Kung] on Device with %d threads per block...:\n", thread_bound_section_length);
			ch8__partial_prefix_sum(Device, {
					.block_dim = {thread_bound_section_length,1,1},
					.kernel_version = CH8__PREFIX_SUM_BRENT_KUNG
			}, 0);

			printf("\nRunning [ch8__partial_prefix_sum for Kogge-Stone/Brent-Kung comparison] on Host...\n");
			ch8__partial_prefix_sum(Host, {}, thread_bound_section_length);

			printf("\nRunning [ch8__partial_prefix_sum 3 phase Kogge-Stone] on Device with %d threads per block and section length equals to %d...:\n", thread_bound_section_length, memory_bound_section_length);
			ch8__partial_prefix_sum(Device, {
					.block_dim = {thread_bound_section_length,1,1},
					.kernel_version = CH8__PREFIX_SUM_3_PHASE_KOGGE_STONE,
					.shared_memory_size = memory_bound_section_size
			}, 0);


			printf("\nRunning [ch8__partial_prefix_sum for 3 phase Kogge-Stone comparison] on Host...\n");
			ch8__partial_prefix_sum(Host, {}, memory_bound_section_length);


			option = -1;
			break;
		case 2:

			printf("\nRunning [ch8__full_prefix_sum Hierarchical 3 phase Kogge-Stone] on Device with %d threads per block and section length equals to %d...:\n", thread_bound_section_length, memory_bound_section_length);
			ch8__full_prefix_sum(Device, {
					.block_dim = {thread_bound_section_length,1,1},
					.kernel_version = CH8__HIERARCHICAL_PREFIX_SUM_3_PHASE_KOGGE_STONE,
					.shared_memory_size = memory_bound_section_size
			});

			printf("\nRunning [ch8__full_prefix_sum single-pass Kogge-Stone] on Device with %d threads per block:\n", thread_bound_section_length);
			ch8__full_prefix_sum(Device, {
					.block_dim = {thread_bound_section_length,1,1},
					.kernel_version = CH8__SINGLE_PASS_PREFIX_SUM_KOGGE_STONE,
					.shared_memory_size = thread_bound_section_length*sizeof(double)
			});

			printf("\nRunning [ch8__full_prefix_sum] on Host...\n");
			ch8__full_prefix_sum(Host, {});

			option = -1;
			break;
		default:
			printf("\t\t[1] - Partial prefix sum (scan by block/section)\n");
			printf("\t\t[2] - Full prefix sum (scan on entire array)\n");
			printf("\nPress the number of the algorithm or zero to go back.\n");
			scanf("%d", &option);
			setbuf(stdin, NULL);
		}
	}
}

static inline void chapter_9_menu(){
	printf("Chapter 09\n");
	printf("Running [ch9__parallel_histogram with block partitioning] on Device:\n");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_WITH_BLOCK_PARTITIONING});

	printf("Running [ch9__parallel_histogram with interleaved partitioning] on Device:\n");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_WITH_INTERLEAVED_PARTITIONING});

	printf("Running [ch9__parallel_histogram with privatization] on Device:\n");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_PRIVATIZED});

	printf("Running [ch9__parallel_histogram with aggregation] on Device:\n");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_AGGREGATED});

	printf("\nRunning [ch9__parallel_histogram] on Host...\n");
	ch9__parallel_histogram(Host, {});
}

static inline void chapter_10_menu(){
	printf("Chapter 10\n");
	printf("Running [ch10__spmv CSR] on Device:\n");
	ch10__spmv(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH10__SPMV_CSR});

	printf("Running [ch10__spmv ELL] on Device:\n");
	ch10__spmv(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH10__SPMV_ELL});


	printf("\nRunning [ch10__spmv CSR] on Host...\n");
	ch10__spmv(Host, {});
}

static inline void chapter_11_menu(){
	printf("Chapter 11\n");
	printf("Running [ch11__merge_sort Basic] on Device:\n");
	ch11__merge_sort(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH11__BASIC_MERGE_SORT});

	printf("Running [ch11__merge_sort tiled] on Device:\n");
	ch11__merge_sort(Device, {.block_dim = {256}, .kernel_version = CH11__TILED_MERGE_SORT});

	printf("Running [ch11__merge_sort circular buffer] on Device:\n");
	ch11__merge_sort(Device, {.block_dim = {256, 1, 1}, .kernel_version = CH11__CIRCULAR_BUFFER_MERGE_SORT});


	printf("\nRunning [ch11__merge_sort] on Host...\n");
	ch11__merge_sort(Host, {});
}



int main(void){

	int main = -1;

	while(main != 0){
		switch(main){
		case 2:
			chapter_2_menu();
			main = -1;
			break;
		case 3:
			chapter_3_menu();
			main = -1;
			break;
		case 4:
			chapter_4_menu();
			main = -1;
			break;
		case 5:
			chapter_5_menu();
			main = -1;
			break;
		case 7:
			chapter_7_menu();
			main = -1;
			break;
		case 8:
			chapter_8_menu();
			main = -1;
			break;
		case 9:
			chapter_9_menu();
			main = -1;
			break;
		case 10:
			chapter_10_menu();
			main = -1;
			break;
		case 11:
			chapter_11_menu();
			main = -1;
			break;
		default:
			printf("\nCHAPTERS:\n");
			printf("\t[Chapter 02] - Data parallel computing (vector addition)\n");
			printf("\t[Chapter 03] - Scalable parallel execution (Image Grayscale and Blur)\n");
			printf("\t[Chapter 04] - Memory and data locality (Matrix Multiplication)\n");
			printf("\t[Chapter 05] - Performance considerations (Array reduction)\n");
			printf("\t[Chapter 07] - Parallel patterns: convolution (1D and 2D convolution)\n");
			printf("\t[Chapter 08] - Parallel patterns: prefix sum (Sequential, Kogge-Stone and Brent-Kung versions)\n");
			printf("\t[Chapter 09] - Parallel patterns: parallel histogram computation\n");
			printf("\t[Chapter 10] - Parallel patterns: sparse matrix computation\n");
			printf("\t[Chapter 11] - Parallel patterns: merge sort\n");
			printf("\nPress the chapter number or zero to exit.\n");
			scanf("%d", &main);
			setbuf(stdin, NULL);
		}
	}




	return 0;
}

