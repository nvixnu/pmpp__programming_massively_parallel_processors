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
#include "chapter_2.h"
#include "nvixnu__array_utils.h" //Map and print functions
#include "nvixnu__error_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__axpy.h"


void ch2__vec_add_device(double *h_x, double *h_y, kernel_config_t config){
	// Pointers to device arrays
	double *d_x, *d_y;

	//Allocates the global memory
	CCE(cudaMalloc(&d_x, N*sizeof(double)));
	CCE(cudaMalloc(&d_y, N*sizeof(double)));

	//Copies the arrays to GPU
	CCE(cudaMemcpy(d_x, h_x, N*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_y, h_y, N*sizeof(double), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	nvixnu__axpy_kernel<<<ceil(N/(config.block_dim.x*1.0)), config.block_dim.x>>>(1.0, d_x, d_y, N);
	CCLE();
	DEVICE_TOC(0);

	//Copies the result back to the heap
	CCE(cudaMemcpy(h_y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost));


	printf("Last %d values:\n", PRINT_SIZE);
	nvixnu__array_map(h_y + (N - PRINT_SIZE), sizeof(double), PRINT_SIZE, nvixnu__print_item_double);


	CCE(cudaFree(d_x));
	CCE(cudaFree(d_y));
}

void ch2__vec_add_host(double *x, double *y){
	HOST_TIC(0);
	nvixnu__axpy_host(1.0, x, y, N);
	HOST_TOC(0);

	printf("Last %d values:\n", PRINT_SIZE);
	nvixnu__array_map(y + (N - PRINT_SIZE), sizeof(double), PRINT_SIZE, nvixnu__print_item_double);
}

void ch2__vec_add(env_e env, kernel_config_t config){
	// Pointers to host arrays
	double *x, *y;

	//Allocates the heap memory
	x = (double*)malloc(N*sizeof(double));
	y = (double*)malloc(N*sizeof(double));

	//Populates the arrays
	nvixnu__populate_multiple_arrays_from_file(FILEPATH, "", "%lf,", "", N, sizeof(double), 2, x, y);

	if(env == Host){
		ch2__vec_add_host(x, y);
	}else{
		ch2__vec_add_device(x, y, config);
	}

	free(x);
	free(y);
}
