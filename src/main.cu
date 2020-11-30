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
#include "utils.h"

int main(void){

	ch2__vec_add({.env = Device});
	ch2__vec_add({.env = Host});

	return 0;
}

