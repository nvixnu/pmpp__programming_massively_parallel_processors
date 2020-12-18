/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 2
 * In this chapter the vector addition and the error handlers functions are presented.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 27/11/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <math.h>
#include <time.h>
#include "ch2__config.h"
#include "nvixnu__array_utils.h" //Map and print functions
#include "nvixnu__error_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__axpy.h"


void ch2__vec_add_device(double *h_x, double *h_y, const int length, kernel_config_t config){
	// Pointers to device arrays
	double *d_x, *d_y;

	//Allocates the global memory
	CCE(cudaMalloc(&d_x, length*sizeof(double)));
	CCE(cudaMalloc(&d_y, length*sizeof(double)));

	//Copies the arrays to GPU
	CCE(cudaMemcpy(d_x, h_x, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_y, h_y, length*sizeof(double), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	nvixnu__axpy_kernel<<<ceil(length/(double)config.block_dim.x), config.block_dim.x>>>(1.0, d_x, d_y, length);
	CCLE();
	DEVICE_TOC(0);

	//Copies the result back to the heap
	CCE(cudaMemcpy(h_y, d_y, length*sizeof(double), cudaMemcpyDeviceToHost));


	CCE(cudaFree(d_x));
	CCE(cudaFree(d_y));
}

void ch2__vec_add_host(double *x, double *y, const int length){
	HOST_TIC(0);
	nvixnu__axpy_host(1.0, x, y, length);
	HOST_TOC(0);
}

void ch2__vec_add(env_e env, kernel_config_t config){
	// Pointers to host arrays
	double *x, *y;

	//Allocates the heap memory
	x = (double*)malloc(CH2__ARRAY_LENGTH*sizeof(double));
	y = (double*)malloc(CH2__ARRAY_LENGTH*sizeof(double));

	//Populates the arrays
	nvixnu__populate_multiple_arrays_from_file(CH2__FILEPATH, "", "%lf,", "", CH2__ARRAY_LENGTH, sizeof(double), 2, x, y);

	if(env == Host){
		ch2__vec_add_host(x, y, CH2__ARRAY_LENGTH);
	}else{
		ch2__vec_add_device(x, y, CH2__ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(y + (CH2__ARRAY_LENGTH - PRINT_LENGTH), sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(x);
	free(y);
}
