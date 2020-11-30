/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 2
 * In this chapter the vector addition and the error handlers functions are presented.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 27/11/2020
 *  Author: Nvixnu
 */

#include "chapter_2.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info
#include "nvixnu__array_utils.h" //Map and print functions
#include "nvixnu__error_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__axpy.h"


void ch2__vec_add_host(){
	// Pointers to host arrays
	double *x, *y;

	// Time handlers
	float duration;
	struct timespec start, stop;

	//Allocates the heap memory
	x = (double*)malloc(N*sizeof(double));
	y = (double*)malloc(N*sizeof(double));

	//Populates the arrays
	nvixnu__populate_multiple_arrays_from_file(FILEPATH, "", "%lf,", "", N, sizeof(double), 2, x, y);

	//Run the host function
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	nvixnu__axpy_host(1.0, x, y, N);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

	//Prints the host result and the elapsed time
	duration = HOST_DURATION_MS(start, stop);
	printf("\nHost elapsed time: %lf ms\n", duration);
	printf("Last %d values:\n", PRINT_SIZE);
	nvixnu__array_map(y + (N - PRINT_SIZE), sizeof(double), PRINT_SIZE, nvixnu__print_item_double);

	free(x);
	free(y);
}

void ch2__vec_add_device(const int block_dim){
	// Pointers to host arrays
	double *h_x, *h_y;
	// Pointers to device arrays
	double *d_x, *d_y;

	// Device time handlers
	float duration;
	cudaEvent_t start, stop;
	CCE(cudaEventCreate(&start));
	CCE(cudaEventCreate(&stop));


	//Allocates the heap memory
	h_x = (double*)malloc(N*sizeof(double));
	h_y = (double*)malloc(N*sizeof(double));

	//Populates the arrays
	nvixnu__populate_multiple_arrays_from_file(FILEPATH, "", "%lf,", "", N, sizeof(double), 2, h_x, h_y);

	//Allocates the global memory
	CCE(cudaMalloc(&d_x, N*sizeof(double)));
	CCE(cudaMalloc(&d_y, N*sizeof(double)));

	//Copies the arrays to GPU
	CCE(cudaMemcpy(d_x, h_x, N*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_y, h_y, N*sizeof(double), cudaMemcpyHostToDevice));

	//Launches the kernel
	CCE(cudaEventRecord(start));
	nvixnu__axpy_kernel<<<ceil(N/(block_dim*1.0)), block_dim>>>(1.0, d_x, d_y, N);
	CCLE();
	CCE(cudaEventRecord(stop));

	//Calculates the elapsed time
	CCE(cudaEventSynchronize(stop));
	CCE(cudaEventElapsedTime(&duration, start, stop));

	//Copies the result back to the heap
	CCE(cudaMemcpy(h_y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost));

	//Prints the kernel result and the elapsed time
	printf("\nKernel elapsed time: %f ms\n", duration);
	printf("Last %d values:\n", PRINT_SIZE);
	nvixnu__array_map(h_y + (N - PRINT_SIZE), sizeof(double), PRINT_SIZE, nvixnu__print_item_double);


	CCE(cudaFree(d_x));
	CCE(cudaFree(d_y));
	free(h_x);
	free(h_y);
}
