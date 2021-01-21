/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 7
 * In this chapter the convolution operation in 1D and 2D arrays is presented.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 07/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include "ch7__config.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__error_utils.h"
#include "pmpp__convolution.h"


void ch7__1d_convolution_device(double *h_input, double *h_output, const int length, const double *h_mask, const int mask_width, kernel_config_t config){
	double *d_input, *d_output, *d_mask;

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));
	CCE(cudaMalloc(&d_mask, mask_width*sizeof(double)));

	CCE(cudaMemcpy(d_mask, h_mask, mask_width*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(length/(double)block_dim);

	DEVICE_TIC(0);
	pmpp__1d_convolution_kernel<<<grid_dim, block_dim, block_dim*sizeof(double)>>>(d_input, d_output, length, d_mask, mask_width);
	CCLE()
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));

}

void ch7__1d_convolution_host(double *input, double *output, const int length, const double *mask, const int mask_width){
	HOST_TIC(0);
	pmpp__1d_convolution_host(input, output, length, mask, mask_width);
	HOST_TOC(0);
}

void ch7__1d_convolution(env_e env, kernel_config_t config){
	double *input, *output;
	const double mask[CH7__1D_MASK_WIDTH] = {1, 0, -1};

	input = (double *)malloc(CH7__1D_ARRAY_LENGTH*sizeof(double));
	output = (double *)malloc(CH7__1D_ARRAY_LENGTH*sizeof(double));

	nvixnu__populate_array_from_file(CH7__1D_FILEPATH, "%lf,", CH7__1D_ARRAY_LENGTH, sizeof(double), input);

	if(env == Host){
		ch7__1d_convolution_host(input, output, CH7__1D_ARRAY_LENGTH, mask, CH7__1D_MASK_WIDTH);
	}else{
		ch7__1d_convolution_device(input, output, CH7__1D_ARRAY_LENGTH, mask, CH7__1D_MASK_WIDTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + (CH7__1D_ARRAY_LENGTH - PRINT_LENGTH), sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}


int main(){
	printf("Chapter 07\n");
	printf("Array with %d Elements\n", CH7__1D_ARRAY_LENGTH);
	printf("Mask width: %d\n", CH7__1D_MASK_WIDTH);

	printf("\n_____ 1d_convolution _____\n\n");

	printf("Running on Device with 256 threads per block...");
	ch7__1d_convolution(Device, {.block_dim = {256,1,1}});

	printf("\nRunning on Device with 1024 threads per block...");
	ch7__1d_convolution(Device, {.block_dim = {1024,1,1}});

	printf("\n_____ 1d_convolution_CPU _____\n");
	ch7__1d_convolution(Host, {});
	return 0;
}