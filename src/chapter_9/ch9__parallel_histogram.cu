/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 9
 * Presents the parallel histogram with the privatization and aggegation techniques.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <string.h>
#include "ch9__config.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"

void ch9__parallel_histogram_device(double *h_input, double *h_output, const int length, kernel_config_t config){
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

void ch9__parallel_histogram_host(double *input, double *output, const int length){
	HOST_TIC(0);
	HOST_TOC(0)
}

void ch9__parallel_histogram(env_e env, kernel_config_t config){
	double *input, *output;

	input = (double *)malloc(CH9__ARRAY_LENGTH*sizeof(double));
	output = (double *)calloc(CH9__ARRAY_LENGTH, sizeof(double));

	nvixnu__populate_array_from_file(CH9__FILEPATH, "%lf,", CH9__ARRAY_LENGTH, sizeof(double), input);

	if(env == Host){
		ch9__parallel_histogram_host(input, output, CH9__ARRAY_LENGTH);
	}else{
		ch9__parallel_histogram_device(input, output, CH9__ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + CH9__ARRAY_LENGTH - PRINT_LENGTH, sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}
