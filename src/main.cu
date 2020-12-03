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
#include "utils.h"

int main(void){

	int main = -1, ch3 = -1;

	while(main != 0){
		switch(main){
			case 2:
				printf("Chapter 2\n");
				printf("Running [vec_add] on Device with 256 threads per block...:\n");
				ch2__vec_add(Device, {.block_dim = {256,1,1}});
				printf("\nRunning [vec_add] on Device with 1024 threads per block...:\n");
				ch2__vec_add(Device, {.block_dim = {1024,1,1}});
				printf("\nRunning [vec_add] on Host...\n");
				ch2__vec_add(Host, {});
				main = -1;
				break;
			case 3:
				while(ch3 != 0){
					printf("\nCHAPTER 3:\n");
					switch(ch3){
						case 1:
							printf("Running [color_to_grayscale] on Device with 256 threads per block...:\n");
							ch3__color_to_grayscale(Device, {.block_dim = {16,16,1}});
							printf("\nRunning [color_to_grayscale] on Device with 1024 threads per block...:\n");
							ch3__color_to_grayscale(Device, {.block_dim = {32,32,1}});
							printf("\nRunning [color_to_grayscale] on Host...\n");
							ch3__color_to_grayscale(Host, {});
							ch3 = -1;
							break;
						case 2:
							printf("Running [blur] on Device with 256 threads per block...:\n");
							ch3__blur(Device, {.block_dim = {16,16,1}});
							printf("\nRunning [blur] on Device with 1024 threads per block...:\n");
							ch3__blur(Device, {.block_dim = {32,32,1}});
							printf("\nRunning [blur] on Host...\n");
							ch3__blur(Host, {});
							ch3 = -1;
							break;
						default:
							printf("\t\t[1] - Color to grayscale\n");
							printf("\t\t[2] - Blur\n");
							printf("\nPress the number of the algorithm or zero to go back.\n");
							scanf("%d", &ch3);
							setbuf(stdin, NULL);
					}
				}
				main = -1;
				break;
			default:
				printf("\nCHAPTERS:\n");
				printf("\t[Chapter 2] - Data parallel computing\n");
				printf("\t[Chapter 3] - Scalable parallel execution\n");
				printf("\nPress the chapter number or zero to exit.\n");
				scanf("%d", &main);
				setbuf(stdin, NULL);
		}
	}




	return 0;
}

