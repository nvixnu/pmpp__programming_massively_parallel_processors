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

	int chapter = 1;

	while(chapter != 0){
		switch(chapter){
			case 2:
				printf("Chapter 2\n");
				printf("Runnung [vec_add] on Host...\n");
				ch2__vec_add({.env = Host});
				printf("\nRunning [vec_add] on Device...\n");
				ch2__vec_add({.env = Device});
				chapter = 1;
				break;
			case 3:
				printf("Chapter 3\n");
				printf("Running [color_to_grayscale] on Host...\n");
				ch3__color_to_grayscale({.env = Host});
				printf("\nRunning [color_to_grayscale] on Device...\n");
				ch3__color_to_grayscale({.env = Device});
				chapter = 1;
				break;
			default:
				printf("\nCHAPTERS:\n");
				printf("\t[Chapter 2] - Data parallel computing\n");
				printf("\t[Chapter 3] - Scalable parallel execution\n");
				printf("\nPress the chapter number to select or zero to exit.\n");
				scanf("%d", &chapter);
				setbuf(stdin, NULL);
		}
	}




	return 0;
}

