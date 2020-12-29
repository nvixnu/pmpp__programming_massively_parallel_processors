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
void ch8__single_pass_kogge_stone_scan(double *input, double *output, const int length, volatile double *scan_value, unsigned int *flags, unsigned int *block_counter){
	extern __shared__ double section_sums[];
	__shared__ float previous_sum;
	__shared__ int sbid;

	if(threadIdx.x == 0){
		sbid = atomicAdd(block_counter, 1);
	}

	__syncthreads();

	const int bid = sbid;
	const int tid = bid*blockDim.x + threadIdx.x;


	if(tid < length){
		section_sums[threadIdx.x] = input[tid];
	}else{
		section_sums[threadIdx.x] = 0.0;
	}

	unsigned int stride;
	for( stride= 1; stride < blockDim.x; stride *= 2){
		__syncthreads();
		if(threadIdx.x >= stride){
			section_sums[threadIdx.x] += section_sums[threadIdx.x - stride];
		}
	}

	__syncthreads();

	if(threadIdx.x == 0){
		//wait for the previous flag
		while(atomicAdd(&flags[bid], 0) == 0);
		//Reads previous partial sum
		previous_sum = scan_value[bid];
		//propagate partial sum
		scan_value[bid + 1]  = previous_sum + section_sums[blockDim.x -1];

		//memory fence
		__threadfence();
		//Set flag
		atomicAdd(&flags[bid + 1], 1);
	}
	__syncthreads();

	output[tid] = previous_sum + section_sums[threadIdx.x];
}

__global__
void ch8__increment_section(double *base, double *output, const int length){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < length && blockIdx.x > 0){
		output[tid] += base[blockIdx.x -1];
	}
}

__global__
void ch8__3_phase_increment_section(double *base, double * output, const int length, const int section_length){
	int phases_count = ceil(section_length/(double)blockDim.x);

	for(int i = 0; i < phases_count; i++){
		int input_index = blockIdx.x*section_length + i*blockDim.x + threadIdx.x;
		if(input_index  < length && blockIdx.x > 0){
			output[input_index] += base[blockIdx.x - 1];
		}
	}
}

void ch8__full_prefix_sum_device(double *h_input, double *h_output, const int length, kernel_config_t config){
	double *d_input, *d_output;
	//Used to holds the sum values of each block in hierarchical parallel method and single-pass scan method
	double *d_block_sum;
	volatile double *d_block_sum_volatile;
	//Used to holds the flags to the adjacent block synchronization in the single-pass scan method
	unsigned int *d_flags;
	//Counter used by dynamic block index assignment in the single-pass scan method
	unsigned int *d_block_counter;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(length/(double)block_dim);

	// Calculates the kernel configuration for the second level/step of the hierarchical method
	const int block_dim_step_2 = grid_dim >= 1024 ? 1024 : (ceil(grid_dim/32.0)*32);
	const int grid_dim_step_2 = ceil(grid_dim/(double)block_dim_step_2);

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));
	CCE(cudaMalloc(&d_block_sum, grid_dim*sizeof(double)));
	CCE(cudaMalloc(&d_block_sum_volatile, grid_dim*sizeof(double)));
	CCE(cudaMalloc(&d_flags, (grid_dim + 1)*sizeof(unsigned int)));
	CCE(cudaMalloc(&d_block_counter, sizeof(unsigned int)));

	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, length*sizeof(double), cudaMemcpyHostToDevice));

	//Initialize the flags, block_counter and the block sums
	CCE(cudaMemset(d_flags, 1, sizeof(unsigned int)));
	CCE(cudaMemset(d_flags + 1, 0, grid_dim*sizeof(unsigned int)));
	CCE(cudaMemset(d_block_counter, 0, sizeof(unsigned int)));
	CCE(cudaMemset((void *)d_block_sum_volatile, 0.0, grid_dim*sizeof(double)));


	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH8__HIERARCHICAL_PREFIX_SUM_3_PHASE_KOGGE_STONE)){
		const int section_length = config.shared_memory_size/sizeof(double);
		const int grid_dim_3_phase = ceil(length/(double)section_length); //The grid_dim is specified according to the shared memory instead of block_dim
		const int grid_dim_3_phase_step_2 = ceil(grid_dim_3_phase/(double)section_length);
		nvixnu__3_phase_kogge_stone_scan_by_block_kernel<<<grid_dim_3_phase, block_dim, config.shared_memory_size>>>(d_input, d_output, length, section_length, d_block_sum);
		CCLE();
		CCE(cudaDeviceSynchronize());
		nvixnu__3_phase_kogge_stone_scan_by_block_kernel<<<grid_dim_3_phase_step_2, block_dim, config.shared_memory_size>>>(d_block_sum, d_block_sum, grid_dim_3_phase, section_length, NULL);
		CCLE();
		CCE(cudaDeviceSynchronize());
		ch8__3_phase_increment_section<<<grid_dim_3_phase, block_dim>>>(d_block_sum, d_output, length, section_length);
		CCLE();
	}else if(!strcmp(config.kernel_version, CH8__SINGLE_PASS_PREFIX_SUM_KOGGE_STONE)){
		ch8__single_pass_kogge_stone_scan<<<grid_dim, block_dim, config.shared_memory_size>>>(d_input, d_output, length, d_block_sum_volatile, d_flags, d_block_counter);
	}else if(!strcmp(config.kernel_version, CH8__SINGLE_PASS_PREFIX_SUM_3_PHASE_KOGGE_STONE)){

	}else{
		printf("\nINVALID KERNEL VERSION\n");
		exit(1);
	}


	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_block_sum));
	CCE(cudaFree((void *)d_block_sum_volatile));
	CCE(cudaFree(d_flags));
	CCE(cudaFree(d_block_counter));
	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));

}

void ch8__full_prefix_sum_host(double *input, double *output, const int length){
	HOST_TIC(0);
	nvixnu__partial_prefix_sum_unit(input, output, length, 1);
	HOST_TOC(0)
}


void ch8__full_prefix_sum(env_e env, kernel_config_t config){
	double *input, *output;

	input = (double *)malloc(CH8__ARRAY_LENGTH*sizeof(double));
	output = (double *)calloc(CH8__ARRAY_LENGTH, sizeof(double));

	nvixnu__populate_array_from_file(CH8__FILEPATH, "%lf,", CH8__ARRAY_LENGTH, sizeof(double), input);

//	for(int i = 0; i < CH8__ARRAY_LENGTH; i++){
//		input[i] = i;
//	}

	if(env == Host){
		ch8__full_prefix_sum_host(input, output, CH8__ARRAY_LENGTH);
	}else{
		ch8__full_prefix_sum_device(input, output, CH8__ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + CH8__ARRAY_LENGTH - PRINT_LENGTH, sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);
	//nvixnu__array_map(output, sizeof(double), CH8__ARRAY_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}

