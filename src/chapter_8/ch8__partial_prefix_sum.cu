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

__host__
__device__
__attribute__((always_inline))
inline void ch8__partial_prefix_sum_unit(double *input, double *output, const int length, const int stride){
	double acc = input[stride - 1];
	output[0] = acc;
	for(int i = stride; i < length; i=i+stride){
		acc += input[i];
		output[i] = acc;
	}
}

__global__
void ch8__kogge_stone_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum){
	extern __shared__ double section_sums[];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	if(tid < length){
		section_sums[threadIdx.x] = input[tid];
	}

	unsigned int stride;
	for( stride= 1; stride < blockDim.x; stride *= 2){
		__syncthreads();
		if(threadIdx.x >= stride){
			section_sums[threadIdx.x] += section_sums[threadIdx.x - stride];
		}
	}
	output[tid] = section_sums[threadIdx.x];
	if(last_sum != NULL && threadIdx.x == (blockDim.x - 1)){
		last_sum[blockIdx.x] = section_sums[threadIdx.x];
	}
}

__global__
void ch8__brent_kung_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum){
	extern __shared__ double section_sums[];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if(tid < length){
		section_sums[threadIdx.x] = input[tid];
	}

	__syncthreads();


	for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
		__syncthreads();
		int idx = (threadIdx.x + 1) * 2 * stride - 1;
		if(idx < blockDim.x){
			section_sums[idx] += section_sums[idx - stride];
		}
	}

	for(int stride = blockDim.x/4; stride > 0; stride /=2){
		__syncthreads();
		int idx = (threadIdx.x + 1) * 2 *stride - 1;
		if((idx + stride) < blockDim.x){
			section_sums[idx + stride] += section_sums[idx];
		}
	}
	__syncthreads();

	output[tid] = section_sums[threadIdx.x];
	if(last_sum != NULL && threadIdx.x == (blockDim.x - 1)){
		last_sum[blockIdx.x] = section_sums[threadIdx.x];
	}
}

__global__
void ch8__kogge_stone_3_phase_scan_by_block_kernel(double *input, double *output, const int length, const int section_length, double *last_sum){
	extern __shared__ double section_sums[];
	int b_dim = blockDim.x;

	// How many phases we should have in order to load the input array to shared memory in a coalesced manner (corner turning)
	int phases_count = ceil(section_length/(double)b_dim);
	// The subsection length is setted to be equals to the phases_count, in order to use all threads in the subsection scan
	int sub_section_max_length = phases_count;


	// Phase 1: Corner turning to load the input data into shared memory
	for(int i = 0; i < phases_count; i++){
		int shared_mem_index = i*b_dim + threadIdx.x;
		int input_index = blockIdx.x*section_length + shared_mem_index;
		//This comparison could be removed if we handle the last phase separately and using the dynamic blockIndex assignment
		if(input_index < length && shared_mem_index < section_length){
			section_sums[shared_mem_index] = input[input_index];
		}
	}

	__syncthreads();

	//Phase 1: Perform the scan on each sub_section
	for(int i = 1; i < sub_section_max_length; i++){
		int index = threadIdx.x*sub_section_max_length + i;
		if(index < section_length){
			section_sums[index] += section_sums[index -1];
		}
	}

	__syncthreads();


	//Phase 2: Performs the Kogge-Stone scan for the last element of each subsection. This step could be performed also by Brent-Kung scan
	for(int stride= 1; stride < section_length; stride *= 2){
		__syncthreads();
		// sub_section_length*threadIdx.x: Indicates the start position of each subsection
		// sub_section_length -1: The last item in a given subsection
		int last_element = sub_section_max_length*threadIdx.x + sub_section_max_length -1;
		if(threadIdx.x >= stride && last_element < section_length){
			section_sums[last_element] += section_sums[last_element - stride*sub_section_max_length];
		}
	}




	__syncthreads();

	//Phase 3: Adding the last element of previous sub_section
	for(int i = 0; i < sub_section_max_length - 1; i++){
		__syncthreads();
		if(threadIdx.x != 0){
			int index = threadIdx.x*sub_section_max_length + i;
			if(index < section_length){
				section_sums[index] += section_sums[threadIdx.x*sub_section_max_length - 1];
			}
		}
	}

	//Save the data on the output array
	for(int i = 0; i < phases_count; i++){
		int output_index = blockIdx.x*section_length + i*b_dim + threadIdx.x;
		if(i*b_dim + threadIdx.x < section_length){
			output[output_index] = section_sums[i*b_dim + threadIdx.x];
		}
	}


}


/**
 * This partial (or sectioned) host version is only for comparison purpose with the partial scan kernels
 */
void ch8__partial_prefix_sum_host(double *input, double *output, const int length, const int stride, const int section_length){
	const int regular_sections_count = length/section_length;
	const int regular_sections_length = regular_sections_count*section_length;
	const int last_section_length = length - regular_sections_length;
	HOST_TIC(0);
	for(int i = 0; i < regular_sections_count; i++){
		ch8__partial_prefix_sum_unit(input + i*section_length, output + i*section_length, section_length, stride);
	}
	if(last_section_length > 0){
		ch8__partial_prefix_sum_unit(input + regular_sections_length, output + regular_sections_length, last_section_length, stride);
	}

	HOST_TOC(0)
}


void ch8__partial_prefix_sum_device(double *h_input, double *h_output, const int length, kernel_config_t config){
	double *d_input, *d_output;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(length/(double)block_dim);
	const int shared_memory = block_dim*sizeof(double);

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));

	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, length*sizeof(double), cudaMemcpyHostToDevice));


	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH8__PREFIX_SUM_KOGGE_STONE)){
		ch8__kogge_stone_scan_by_block_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, length, NULL);
	}else if(!strcmp(config.kernel_version, CH8__PREFIX_SUM_BRENT_KUNG)){
		ch8__brent_kung_scan_by_block_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, length, NULL);
	}else if(!strcmp(config.kernel_version, CH8__PREFIX_SUM_KOGGE_STONE_3_PHASE)){
		const int buffer_length = config.shared_memory_length/sizeof(double);
		const int grid_dim_3_phase = ceil(length/(double)buffer_length); //The grid_dim is specified according to the shared memory instead of block_dim
		ch8__kogge_stone_3_phase_scan_by_block_kernel<<<grid_dim_3_phase, block_dim, config.shared_memory_length>>>(d_input, d_output, length, buffer_length, NULL);
	}else{
		printf("\nINVALID KERNEL VERSION\n");
	}
	CCLE();

	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));


	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));

}


void ch8__partial_prefix_sum(env_e env, kernel_config_t config, const int section_length){
	double *input, *output;

	input = (double *)malloc(CH8__ARRAY_LENGTH_FOR_PARTIAL_SCAN*sizeof(double));
	output = (double *)calloc(CH8__ARRAY_LENGTH_FOR_PARTIAL_SCAN, sizeof(double));

	nvixnu__populate_array_from_file(CH8__FILEPATH, "%lf,", CH8__ARRAY_LENGTH_FOR_PARTIAL_SCAN, sizeof(double), input);


	if(env == Host){
		ch8__partial_prefix_sum_host(input, output, CH8__ARRAY_LENGTH_FOR_PARTIAL_SCAN, 1, section_length);
	}else{
		ch8__partial_prefix_sum_device(input, output, CH8__ARRAY_LENGTH_FOR_PARTIAL_SCAN, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + CH8__ARRAY_LENGTH_FOR_PARTIAL_SCAN - PRINT_LENGTH, sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(input);
	free(output);

	return;
}
