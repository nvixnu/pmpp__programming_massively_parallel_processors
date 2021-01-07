/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 8
 * Presents the prefix sum algorithm by the Kogge-Stone and Brent-King designs
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 8/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ch8__config.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"
#include "pmpp__prefix_sum.h"


/**
 * This partial (or sectioned) host version is only for comparison purpose with the partial scan kernels
 */
void ch8__partial_prefix_sum_host(double *input, double *output, const int length, const int stride, const int section_length){
	const int regular_sections_count = length/section_length;
	const int regular_sections_length = regular_sections_count*section_length;
	const int last_section_length = length - regular_sections_length;
	HOST_TIC(0);
	for(int i = 0; i < regular_sections_count; i++){
		pmpp__partial_prefix_sum_unit(input + i*section_length, output + i*section_length, section_length, stride);
	}
	if(last_section_length > 0){
		pmpp__partial_prefix_sum_unit(input + regular_sections_length, output + regular_sections_length, last_section_length, stride);
	}

	HOST_TOC(0)
}


void ch8__partial_prefix_sum_device(double *h_input, double *h_output, const int length, kernel_config_t config){
	double *d_input, *d_output;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(length/(double)block_dim);
	size_t shared_memory = block_dim*sizeof(double);

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));

	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, length*sizeof(double), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH8__PREFIX_SUM_KOGGE_STONE)){
		pmpp__kogge_stone_scan_by_block_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, length, NULL);
	}else if(!strcmp(config.kernel_version, CH8__PREFIX_SUM_BRENT_KUNG)){
		pmpp__brent_kung_scan_by_block_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, length, NULL);
	}else if(!strcmp(config.kernel_version, CH8__PREFIX_SUM_3_PHASE_KOGGE_STONE)){
		const int buffer_length = config.shared_memory_size/sizeof(double);
		const int grid_dim_3_phase = ceil(length/(double)buffer_length); //The grid_dim is specified according to the shared memory instead of block_dim
		pmpp__3_phase_kogge_stone_scan_by_block_kernel<<<grid_dim_3_phase, block_dim, config.shared_memory_size>>>(d_input, d_output, length, buffer_length, NULL);
	}else{
		printf("\nINVALID KERNEL VERSION\n");
		exit(1);
	}
	CCLE();

	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));


	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));

}


void ch8__partial_prefix_sum(env_e env, kernel_config_t config, const int section_length){
	double *input, *output;

	input = (double *)malloc(CH8__ARRAY_LENGTH*sizeof(double));
	output = (double *)calloc(CH8__ARRAY_LENGTH, sizeof(double));

	nvixnu__populate_array_from_file(CH8__FILEPATH, "%lf,", CH8__ARRAY_LENGTH, sizeof(double), input);


	if(env == Host){
		ch8__partial_prefix_sum_host(input, output, CH8__ARRAY_LENGTH, 1, section_length);
	}else{
		ch8__partial_prefix_sum_device(input, output, CH8__ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + CH8__ARRAY_LENGTH - PRINT_LENGTH, sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}
