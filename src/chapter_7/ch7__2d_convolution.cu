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
#include "chapter_7.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__error_utils.h"
#include "nvixnu__convolution.h"


void ch7__2d_convolution_device(double *h_input, double *h_output, const int width, const int height, const double *h_mask, const int mask_width, kernel_config_t config){
	double *d_input, *d_output, *d_mask;

	const int length = width * height;
	const int mask_length = mask_width*mask_width;

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));
	CCE(cudaMalloc(&d_mask, mask_length*sizeof(double)));

	CCE(cudaMemcpy(d_mask, h_mask, mask_length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));


    const int ghosts = mask_width -1;
    const int padded_block_dim_x = config.block_dim.x + ghosts;
    const int padded_block_dim_y = config.block_dim.y + ghosts;
    const int shared_memory = padded_block_dim_x*padded_block_dim_y*sizeof(double);
    dim3 block_dim(config.block_dim.x, config.block_dim.y, 1);
    dim3 grid_dim(ceil(width/(double)config.block_dim.x), ceil(height/(double)config.block_dim.y), 1);

	DEVICE_TIC(0);
	nvixnu__2d_convolution_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, width, height, d_mask, mask_width);
	CCLE()
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_input));
	CCE(cudaFree(d_mask));
	CCE(cudaFree(d_output));

}

void ch7__2d_convolution_host(double *input, double *output, const int width, const int height, const double *mask, const int mask_width){
	HOST_TIC(0);
	nvixnu__2d_convolution_host(input, output, width, height, mask, mask_width);
	HOST_TOC(0);
}

void ch7__2d_convolution(env_e env, kernel_config_t config){
	double *input, *output;

	input = (double *)malloc(CH7__2D_ARRAY_LENGTH*sizeof(double));
	output = (double *)malloc(CH7__2D_ARRAY_LENGTH*sizeof(double));

	nvixnu__populate_array_from_file(CH7__2D_FILEPATH, "%lf,", CH7__2D_ARRAY_LENGTH, sizeof(double), input);

	if(env == Host){
		ch7__2d_convolution_host(input, output, CH7__2D_ARRAY_WIDTH, CH7__2D_ARRAY_HEIGHT, H_MASK, CH7__1D_MASK_WIDTH);
	}else{
		ch7__2d_convolution_device(input, output, CH7__2D_ARRAY_WIDTH, CH7__2D_ARRAY_HEIGHT, H_MASK, CH7__1D_MASK_WIDTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + (CH7__2D_ARRAY_LENGTH - PRINT_LENGTH), sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}
