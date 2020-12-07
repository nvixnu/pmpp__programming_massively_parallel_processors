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
#include "chapter_5.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"
#include "nvixnu__reduction.h"


double ch5__sum_reduction_device(double *h_v, const int length, kernel_config_t config){
	double *d_v, h_sum;
	int old_grid_dim;
	int grid_dim = ceil(length/(double)config.block_dim.x);

	CCE(cudaMalloc(&d_v, length*sizeof(double)));

	CCE(cudaMemcpy(d_v, h_v, length*sizeof(double), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	while(grid_dim > 1){ // If addtional reductions are necessary (SIZE > BLOCK_DIM)
		old_grid_dim = grid_dim;
		grid_dim = ceil(grid_dim/(double)config.block_dim.x); // The array d_sum has grid_dim elements, so instead SIZE, we use GRID_DIM
		nvixnu__sum_by_block_kernel<<<grid_dim, config.block_dim.x, config.block_dim.x*sizeof(double)>>>(d_v, d_v, old_grid_dim);
		CCLE();
		CCE(cudaDeviceSynchronize());
	}
	DEVICE_TOC(0);


	CCE(cudaMemcpy(&h_sum, d_v, sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_v));

	return h_sum;

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

	v = (double *)malloc(CREDIT_CARD_DATASET_LENGTH*sizeof(double));

	nvixnu__populate_array_from_file(CREDIT_CARD_DATASET_PATH, "%lf,", CREDIT_CARD_DATASET_LENGTH, sizeof(double), v);

	if(env == Host){
		sum = ch5__sum_reduction_host(v, CREDIT_CARD_DATASET_LENGTH);
	}else{
		sum = ch5__sum_reduction_device(v, CREDIT_CARD_DATASET_LENGTH, config);
	}

	printf("The sum is %f\n", sum);

	free(v);

	return;
}
