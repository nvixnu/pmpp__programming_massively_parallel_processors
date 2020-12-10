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
#include "nvixnu__cuda_devices_props.h"

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
void ch8__brent_kung_3_phase_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum){
	extern __shared__ double section_sums[];
	int b_dim = blockDim.x;
	int tx = threadIdx.x;
	int tid = blockIdx.x*b_dim + tx;
	int phases = ceil(length/(double)b_dim);


	if(tid < length){
		for(int p = 0; p < phases; p++){
			section_sums[p*b_dim + tx] = input[tid];
		}
	}
	__syncthreads();

	double *start;

	start = section_sums + (tx*phases);

	ch8__partial_prefix_sum_unit(start, start, phases, 1);

	__syncthreads();
	if(tx == 0){
		ch8__partial_prefix_sum_unit(start, start, length, phases);
	}

	__syncthreads();

	if(tid < length){
		output[tx] = section_sums[tx]; //Section zero
		for(int p = 1; p < phases; p++){
			output[p*b_dim + tx] = section_sums[p*b_dim + tx] + section_sums[(p-1)*b_dim + phases -1];
		}
	}

}


void ch8__partial_prefix_sum_host(double *input, double *output, const int length, const int stride){
	HOST_TIC(0);
	ch8__partial_prefix_sum_unit(input, output, length, stride);
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
	}else if(!strcmp(config.kernel_version, CH8__PREFIX_SUM_BRENT_KUNG_3_PHASE)){
		//cudaDeviceProp device_props =  nvixnu__get_cuda_device_props(0);
		//size_t max_shared_mem = device_props.sharedMemPerBlock;
		//ch8__brent_kung_3_phase_scan_by_block_kernel<<<grid_dim, block_dim, three_phase_shared_memory>>>(d_input, d_output, length, d_block_sum);
	}else{
		printf("\nINVALID KERNEL VERSION\n");
	}
	CCLE();

	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));


	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));

}


void ch8__partial_prefix_sum(env_e env, kernel_config_t config){
	double *input, *output;

	input = (double *)malloc(CH8__ARRAY_LENGTH*sizeof(double));
	output = (double *)calloc(CH8__ARRAY_LENGTH, sizeof(double));

	nvixnu__populate_array_from_file(CH8__FILEPATH, "%lf,", CH8__ARRAY_LENGTH, sizeof(double), input);


	if(env == Host){
		ch8__partial_prefix_sum_host(input, output, CH8__ARRAY_LENGTH, 1);
	}else{
		ch8__partial_prefix_sum_device(input, output, CH8__ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + (CH8__ARRAY_LENGTH - PRINT_LENGTH), sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);



	free(input);
	free(output);

	return;
}
