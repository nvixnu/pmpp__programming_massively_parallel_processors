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
#include "pmpp__histogram.h"


void ch9__parallel_histogram_device(char *h_input, const int input_length, int *h_output, const int output_length , kernel_config_t config){
	char *d_input;
	int *d_output;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(input_length/(double)block_dim);
	const int shared_memory = block_dim * sizeof(char);

	CCE(cudaMalloc(&d_input, input_length*sizeof(char)));
	CCE(cudaMalloc(&d_output, output_length*sizeof(int)));

	CCE(cudaMemcpy(d_input, h_input, input_length*sizeof(char), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, output_length*sizeof(int), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH9__HISTOGRAM_WITH_BLOCK_PARTITIONING)){
		pmpp__histogram_with_block_partitioning_kernel<<<grid_dim, block_dim>>>(d_input, input_length, d_output);
		CCLE();
	}else if(!strcmp(config.kernel_version, CH9__HISTOGRAM_WITH_INTERLEAVED_PARTITIONING)){
		pmpp__histogram_with_interleaved_partitioning_kernel<<<grid_dim, block_dim>>>(d_input, input_length, d_output);
		CCLE();
	}else if(!strcmp(config.kernel_version, CH9__HISTOGRAM_PRIVATIZED)){
		pmpp__histogram_privatized_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, input_length, d_output, output_length);
		CCLE();
	}else if(!strcmp(config.kernel_version, CH9__HISTOGRAM_AGGREGATED)){
		pmpp__histogram_aggregated_kernel<<<grid_dim, block_dim, shared_memory>>>(d_input, input_length, d_output, output_length);
		CCLE();
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
	pmpp__histogram_host(input, length, output);
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

int main(){
	printf("Chapter 09\n");
	printf("Array with %d Elements\n", CH9__INPUT_ARRAY_LENGTH);

	printf("\n_____ parallel_histogram [Block partitioning] _____\n\n");

	printf("Running on Device with 1024 threads per block...");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_WITH_BLOCK_PARTITIONING});

	printf("\n_____ parallel_histogram [Interleaved partitioning] _____\n");
	printf("Running on Device with 1024 threads per block...");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_WITH_INTERLEAVED_PARTITIONING});

	printf("\n_____ parallel_histogram [Privatization] _____\n");
	printf("Running on Device with 1024 threads per block...");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_PRIVATIZED});

	printf("\n_____ parallel_histogram [Aggregation] _____\n");
	printf("Running on Device with 1024 threads per block...");
	ch9__parallel_histogram(Device, {.block_dim = {1024, 1, 1}, .kernel_version = CH9__HISTOGRAM_AGGREGATED});

	printf("\n_____ parallel_histogram_CPU _____\n");
	ch9__parallel_histogram(Host, {});

	return 0;
}
