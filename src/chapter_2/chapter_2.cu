/*
 * chapter_2.cu
 *
 *  Created on: 27/11/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <math.h>

#include "chapter_2.h"

#include <time.h>
#include "nvixnu__cuda_devices_props.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__error_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__axpy.h"

#define INPUT_ROWS 284807
#define INPUT_COLS 28
#define N INPUT_ROWS*INPUT_COLS/2
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define FILENAME "../datasets/credit_card_fraud/" STR(INPUT_ROWS) "x" STR(INPUT_COLS) ".csv"

void ch2__vec_add_device(void){

}

void ch2__vec_add_host(void){

}


void ch2__vec_add_host_x_device(void){
	// Host and device arrays
	double *h_x, *h_y, *d_x, *d_y, a = 1.0;
	// Elapsed time handlers
	struct timespec h_start, h_stop;
	float k_duration, h_duration;
	cudaEvent_t k_start, k_stop;

	// Create the time handlers
	CCE(cudaEventCreate(&k_start));
	CCE(cudaEventCreate(&k_stop));

	//Allocates the heap memory
	h_x = (double*)malloc(N*sizeof(double));
	h_y = (double*)malloc(N*sizeof(double));

	//Populates the arrays
	nvixnu__populate_multiple_arrays_from_file(FILENAME, "", "%lf,", "", N, sizeof(double), 2, h_x, h_y);

	//Allocates the global memory
	CCE(cudaMalloc(&d_x, N*sizeof(double)));
	CCE(cudaMalloc(&d_y, N*sizeof(double)));

	//Copies the arrays to GPU
	CCE(cudaMemcpy(d_x, h_x, N*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_y, h_y, N*sizeof(double), cudaMemcpyHostToDevice));

	//Launches the kernel
	CCE(cudaEventRecord(k_start));
	nvixnu__axpy<<<ceil(N/1024.0), 1024>>>(a, d_x, d_y, N);
	CCLE();
	CCE(cudaEventRecord(k_stop));

	//Calculates the elapsed time
	CCE(cudaEventSynchronize(k_stop));
	CCE(cudaEventElapsedTime(&k_duration, k_start, k_stop));

	//Copies the result back to the heap
	CCE(cudaMemcpy(h_y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost));


	//Prints the result
	printf("\nKernel elapsed time: %f ms\n", k_duration);
	printf("Last 20 values:\n");
	nvixnu__array_map(h_y + (N - 20), sizeof(double), 20, nvixnu__print_item_double);


	//Re-populates the h_y array
	nvixnu__populate_multiple_arrays_from_file(FILENAME, "%*lf, ", "%lf,", "", N, sizeof(double), 1, h_y);

	//Run the host function
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &h_start);
	nvixnu__h_axpy(a, h_x, h_y, N);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &h_stop);

	h_duration = (h_stop.tv_sec - h_start.tv_sec) * 1e3 + (h_stop.tv_nsec - h_start.tv_nsec) / 1e6;
	printf("\nHost elapsed time: %lf ms\n", h_duration);
	printf("Last 20 values:\n");
	nvixnu__array_map(h_y + (N - 20), sizeof(double), 20, nvixnu__print_item_double);

	CCE(cudaFree(d_x));
	CCE(cudaFree(d_y));
	free(h_x);
	free(h_y);
}
