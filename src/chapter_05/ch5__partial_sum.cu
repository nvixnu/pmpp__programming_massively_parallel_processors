/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 5
 * In this chapter the partial vector sum (reduction) is presented.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 06/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <math.h>
#include "ch5__config.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"
#include "nvixnu__reduction.h"


double ch5__sum_reduction_device(double *h_v, const int length, kernel_config_t config){
	double *d_v, *h_partial_sum, *d_partial_sum, h_final_sum;

	// Colculates the initial grid_dim and block_dim values
	int old_grid_dim;
	int block_dim = config.block_dim.x;
	int grid_dim = ceil(length/(double)block_dim);

	h_partial_sum = (double *)calloc(grid_dim, sizeof(double)); //The size of the "sum" arrays is the grid dimension, once the partial sums is performed  one per block


	CCE(cudaMalloc(&d_v, length*sizeof(double)));
	CCE(cudaMalloc(&d_partial_sum, grid_dim*sizeof(double)));

	CCE(cudaMemcpy(d_v, h_v, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_partial_sum, h_partial_sum, grid_dim*sizeof(double), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	// Reduces the d_v array (SIZE elements) to d_partial_sum array (with grid_dim elements)
	nvixnu__sum_by_block_kernel<<<grid_dim, block_dim, block_dim*sizeof(double)>>>(d_v, d_partial_sum, length);
	CCLE();
	CCE(cudaDeviceSynchronize());

	while(grid_dim > 1){ // Runs if addtional reductions are necessary (length > block_dim), since the kernel performs the reduction only inside each block
		old_grid_dim = grid_dim;
		grid_dim = ceil(grid_dim/(double)block_dim); // The array d_partial_sum has grid_dim elements, so instead length, we use grid_dim
		nvixnu__sum_by_block_kernel<<<grid_dim, block_dim, block_dim*sizeof(double)>>>(d_partial_sum, d_partial_sum, old_grid_dim);
		CCLE();
		CCE(cudaDeviceSynchronize());
	}
	DEVICE_TOC(0);

	CCE(cudaMemcpy(&h_final_sum, d_partial_sum, sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_v));
	CCE(cudaFree(d_partial_sum));
	free(h_partial_sum);

	return h_final_sum;

}

double ch5__sum_reduction_host(double *v, const int length){
	double sum = 0;

	HOST_TIC(0);
	for(int i = 0; i < length; i++){
		sum += v[i];
	}
	HOST_TOC(0);

	return sum;
}

void ch5__sum_reduction(env_e env, kernel_config_t config){
	double *v, sum;

	v = (double *)malloc(CH5__ARRAY_LENGTH*sizeof(double));

	nvixnu__populate_array_from_file(CH5__FILEPATH, "%lf,", CH5__ARRAY_LENGTH, sizeof(double), v);

	if(env == Host){
		sum = ch5__sum_reduction_host(v, CH5__ARRAY_LENGTH);
	}else{
		sum = ch5__sum_reduction_device(v, CH5__ARRAY_LENGTH, config);
	}

	printf("The sum is %f\n", sum);

	free(v);

	return;
}
