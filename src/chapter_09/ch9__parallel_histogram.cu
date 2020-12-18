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


__global__
void ch9__histogram_aggregated_kernel(char* input, const int input_length, int* output, const int output_length) {
	extern __shared__ unsigned int histo_s[];
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx +=blockDim.x) {
		histo_s[binIdx] = 0u;
	}
	__syncthreads();
	unsigned int prev_index = -1;
	unsigned int accumulator = 0;

	for(unsigned int i = tid; i < input_length; i += blockDim.x*gridDim.x) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			unsigned int curr_index = alphabet_position/4;
			if (curr_index != prev_index) {
				if (accumulator >= 0) atomicAdd(&(histo_s[alphabet_position/4]), accumulator);
				accumulator = 1;
				prev_index = curr_index;
			}
			else {
				accumulator++;
			}
		}
	}
	__syncthreads();

	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx += blockDim.x) {
		atomicAdd(&(output[binIdx]), histo_s[binIdx]);
	}
}


__global__
void ch9__histogram_privatized_kernel(char* input, const int input_length, int* output, const int output_length) {
	extern __shared__ unsigned int histo_s[];

	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;


	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx +=blockDim.x) {
		histo_s[binIdx] = 0;
	}
	__syncthreads();

	for (unsigned int i = tid; i < input_length; i += blockDim.x*gridDim.x) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26){
			atomicAdd(&(histo_s[alphabet_position/4]), 1);
		}
	}
	__syncthreads();
	for(unsigned int binIdx = threadIdx.x; binIdx < output_length; binIdx += blockDim.x) {
		atomicAdd(&(output[binIdx]), histo_s[binIdx]);
	}
}

__global__
void ch9__histogram_with_interleaved_partitioning_kernel(char *input, const int length, int *output){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (unsigned int i = tid; i < length; i += blockDim.x*gridDim.x ) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26){
			atomicAdd(&(output[alphabet_position/4]), 1);
		}
	}
}

__global__
void ch9__histogram_with_block_partitioning_kernel(char *input, const int length, int *output){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int section_size = ceil(length/(double)(blockDim.x * gridDim.x));
	int start = i*section_size;

	for (int k = 0; k < section_size; k++) {
		if (start+k < length) {
			int alphabet_position = input[start+k] - 'a';
			if (alphabet_position >= 0 && alphabet_position < 26){
				atomicAdd(&(output[alphabet_position/4]), 1);
			}
		}
	}
}

void ch9__parallel_histogram_device(char *h_input, const int input_length, int *h_output, const int output_length , kernel_config_t config){
	char *d_input;
	int *d_output;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(input_length/(double)block_dim);

	CCE(cudaMalloc(&d_input, input_length*sizeof(char)));
	CCE(cudaMalloc(&d_output, output_length*sizeof(int)));

	CCE(cudaMemcpy(d_input, h_input, input_length*sizeof(char), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, output_length*sizeof(int), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH9__HISTOGRAM_WITH_BLOCK_PARTITIONING)){
		ch9__histogram_with_block_partitioning_kernel<<<grid_dim, block_dim>>>(d_input, input_length, d_output);
	}else if(!strcmp(config.kernel_version, CH9__HISTOGRAM_WITH_INTERLEAVED_PARTITIONING)){
		ch9__histogram_with_interleaved_partitioning_kernel<<<grid_dim, block_dim>>>(d_input, input_length, d_output);
	}else if(!strcmp(config.kernel_version, CH9__HISTOGRAM_PRIVATIZED)){
		ch9__histogram_privatized_kernel<<<grid_dim, block_dim>>>(d_input, input_length, d_output, output_length);
	}else if(!strcmp(config.kernel_version, CH9__HISTOGRAM_AGGREGATED)){
		ch9__histogram_aggregated_kernel<<<grid_dim, block_dim>>>(d_input, input_length, d_output, output_length);
	}else{
		printf("\nINVALID KERNEL VERSION\n");
		exit(1);
	}
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, output_length*sizeof(int), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));
}

void ch9__parallel_histogram_host(char *input, const int length, int *output){
	HOST_TIC(0);
	for (int i = 0; i < length; i++) {
		int alphabet_position = input[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			output[alphabet_position/4]++;
		}
	}
	HOST_TOC(0)
}

void ch9__parallel_histogram(env_e env, kernel_config_t config){
	char *input;
	int *output;

	input = (char *)malloc(CH9__INPUT_ARRAY_LENGTH*sizeof(char));
	output = (int *)calloc(CH9__OUTPUT_ARRAY_LENGTH, sizeof(int));

	nvixnu__populate_array_from_file(CH9__FILEPATH, "%c", CH9__INPUT_ARRAY_LENGTH, sizeof(char), input);

	if(env == Host){
		ch9__parallel_histogram_host(input, CH9__INPUT_ARRAY_LENGTH, output);
	}else{
		ch9__parallel_histogram_device(input, CH9__INPUT_ARRAY_LENGTH, output, CH9__OUTPUT_ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + CH9__OUTPUT_ARRAY_LENGTH - PRINT_LENGTH, sizeof(int), PRINT_LENGTH, nvixnu__print_item_int);

	free(input);
	free(output);

	return;
}
