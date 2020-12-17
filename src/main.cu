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
#include "chapter_2/ch2__config.h"
#include "chapter_3/ch3__config.h"
#include "chapter_4/ch4__config.h"
#include "chapter_5/ch5__config.h"
#include "chapter_7/ch7__config.h"
#include "chapter_8/ch8__config.h"
#include "utils.h"


static inline void chapter_2_menu(){
	printf("Chapter 2\n");
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
		printf("\nCHAPTER 3:\n");
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
	printf("Chapter 4\n");
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
	printf("Chapter 5\n");
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
		printf("\nCHAPTER 7:\n");
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
		printf("\nCHAPTER 8:\n");
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
		default:
			printf("\nCHAPTERS:\n");
			printf("\t[Chapter 2] - Data parallel computing (vector addition)\n");
			printf("\t[Chapter 3] - Scalable parallel execution (Image Grayscale and Blur)\n");
			printf("\t[Chapter 4] - Memory and data locality (Matrix Multiplication)\n");
			printf("\t[Chapter 5] - Performance considerations (Array reduction)\n");
			printf("\t[Chapter 7] - Parallel patterns: convolution (1D and 2D convolution)\n");
			printf("\t[Chapter 8] - Parallel patterns: prefix sum (Sequantial, Kogge-Stone and Brent-Kung)\n");
			printf("\nPress the chapter number or zero to exit.\n");
			scanf("%d", &main);
			setbuf(stdin, NULL);
		}
	}




	return 0;
}

