/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 11
 * Presents the merge sort algorithm and tiling with dynamic input data identification.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <string.h>
#include "ch11__config.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"

int co_rank(int k, int *A, int m, int *B, int n) {
	int i= k<m ? k : m; //i = min(k,m)
	int j = k- i;
	int i_low = 0>(k-n) ? 0 : k-n; //i_low = max(0, k-n)
	int j_low = 0>(k-m) ? 0 : k-m; //i_low = max(0, k-m)
	int delta;
	bool active = true;
	while(active){
		if (i > 0 && j < n && A[i-1] > B[j]) {
			delta = ((i - i_low +1) >> 1); // ceil(i-i_low)/2)
			j_low = j;
			j = j + delta;
			i = i - delta;
		} else if (j > 0 && i < m && B[j-1] >= A[i]) {
			delta = ((j - j_low +1) >> 1);
			i_low = i;
			i = i + delta;
			j = j - delta;
		}else {
			active = false;
		}
	}
	return i;
}

void ch11__merge_sort_device(double *h_input, double *h_output, const int length, kernel_config_t config){
	double *d_input, *d_output;

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));

	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, length*sizeof(double), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, "XXX")){

	}else{
		printf("\nINVALID KERNEL VERSION\n");
		exit(1);
	}
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));
}

void ch11__merge_sort_host(int *A, int m, int *B, int n, int *C){
	HOST_TIC(0);
	//done with A[] handle remaining B[]
	int i = 0; //index into A 3
	int j = 0; //index into B
	int k = 0; //index into C 13
	// handle the start of A[] and B[]
	while ((i < m) && (j < n)) {
		if (A[i] <= B[j]) {
			C[k++] = A[i++];
		} else {
			C[k++] = B[j++];
		}
	}
	if (i == m) {
		for (; j < n; j++) {
			C[k++] = B[j];
		}
	} else {
		for (; i <m; i++) {
			C[k++] = A[i];
		}

	}
	HOST_TOC(0)
}

void ch11__merge_sort(env_e env, kernel_config_t config){
	double *input, *output;

	input = (double *)malloc(CH9__ARRAY_LENGTH*sizeof(double));
	output = (double *)calloc(CH9__ARRAY_LENGTH, sizeof(double));

	nvixnu__populate_array_from_file(CH9__FILEPATH, "%lf,", CH9__ARRAY_LENGTH, sizeof(double), input);

	if(env == Host){
		ch11__merge_sort_host(input, output, CH9__ARRAY_LENGTH);
	}else{
		ch11__merge_sort_device(input, output, CH9__ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + CH9__ARRAY_LENGTH - PRINT_LENGTH, sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}
