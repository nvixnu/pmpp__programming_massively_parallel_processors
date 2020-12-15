/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 8
 * Presents the prefix sum algorithm by the Kogge-Stone and Brent-King designs for arbitrary input length
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 15/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ch8__config.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"
#include "nvixnu__prefix_sum.h"

__global__
void ch8__increment_section(double *base, double * output, const int length){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < length && blockIdx.x > 0){
		output[tid] += base[blockIdx.x -1];
	}
}

void ch8__full_prefix_sum_device(double *h_input, double *h_output, const int length, kernel_config_t config){
	double *d_input, *d_output, *d_block_sum;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(length/(double)block_dim);
	const int shared_memory = block_dim*sizeof(double);

	const int block_dim_step_2 = grid_dim > 1024 ? 1024 : (ceil(grid_dim/32.0)*32);
	const int grid_dim_step_2 = ceil(grid_dim/(double)block_dim_step_2);
	const int shared_memory_step_2 = block_dim_step_2*sizeof(double);

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));
	CCE(cudaMalloc(&d_block_sum, grid_dim*sizeof(double)));

	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, length*sizeof(double), cudaMemcpyHostToDevice));


	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH8__HIERARCHICAL_PREFIX_SUM_KOGGE_STONE)){
		nvixnu__kogge_stone_scan_by_block_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, length, d_block_sum);
		CCLE();
		CCE(cudaDeviceSynchronize());
		nvixnu__kogge_stone_scan_by_block_kernel<<<grid_dim_step_2, block_dim_step_2, shared_memory_step_2>>>(d_block_sum, d_block_sum, grid_dim, NULL);
		CCLE();
		CCE(cudaDeviceSynchronize());
		ch8__increment_section<<<grid_dim, block_dim>>>(d_block_sum, d_output, length);
		CCLE();
		CCE(cudaDeviceSynchronize());
	}else{
		printf("\nINVALID KERNEL VERSION\n");
	}

	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_block_sum));
	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));

}

void ch8__full_prefix_sum_host(double *input, double *output, const int length){
	HOST_TIC(0);
	nvixnu__partial_prefix_sum_unit(input, output, length, 1);
	HOST_TOC(0)
}


void ch8__full_prefix_sum(env_e env, kernel_config_t config, const int section_length){
	double *input, *output;

	input = (double *)malloc(CH8__ARRAY_LENGTH_FOR_FULL_SCAN*sizeof(double));
	output = (double *)calloc(CH8__ARRAY_LENGTH_FOR_FULL_SCAN, sizeof(double));

	nvixnu__populate_array_from_file(CH8__FILEPATH, "%lf,", CH8__ARRAY_LENGTH_FOR_FULL_SCAN, sizeof(double), input);


	if(env == Host){
		ch8__full_prefix_sum_host(input, output, CH8__ARRAY_LENGTH_FOR_FULL_SCAN);
	}else{
		ch8__full_prefix_sum_device(input, output, CH8__ARRAY_LENGTH_FOR_FULL_SCAN, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + CH8__ARRAY_LENGTH_FOR_FULL_SCAN - PRINT_LENGTH, sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}

