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
#include "chapter_2/chapter_2.h"
#include "chapter_3/chapter_3.h"
#include "chapter_4/chapter_4.h"
#include "chapter_5/chapter_5.h"
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
		default:
			printf("\nCHAPTERS:\n");
			printf("\t[Chapter 2] - Data parallel computing (vector addition)\n");
			printf("\t[Chapter 3] - Scalable parallel execution (Image Grayscale and Blur)\n");
			printf("\t[Chapter 4] - Memory and data locality (Matrix Multiplication)\n");
			printf("\t[Chapter 5] - Performance considerations (Array reduction)\n");
			printf("\nPress the chapter number or zero to exit.\n");
			scanf("%d", &main);
			setbuf(stdin, NULL);
		}
	}




	return 0;
}

