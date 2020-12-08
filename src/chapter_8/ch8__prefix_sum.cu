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
#include <math.h>
#include "chapter_8.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"


__global__
void kogge_stone_scan_by_block(double *input, double *output, const int length, double *last_sum){
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
    __syncthreads();
    output[tid] = section_sums[threadIdx.x];
    if(last_sum != NULL && threadIdx.x == (blockDim.x - 1)){
        last_sum[blockIdx.x] = section_sums[threadIdx.x];
    }
}

__global__
void brent_kung_scan_by_block(double *input, double *output, const int length, double *last_sum){
    extern __shared__ double section_sums[];

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if(tid < length){
        section_sums[threadIdx.x] = input[tid];
    }

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

void ch8__prefix_sum_device(double *h_input, double *h_output, const int length, kernel_config_t config){
	double *d_input, *d_output;

	CCE(cudaMalloc(&d_input, length*sizeof(double)));
	CCE(cudaMalloc(&d_output, length*sizeof(double)));

	CCE(cudaMemcpy(d_input, h_input, length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_output, h_output, length*sizeof(double), cudaMemcpyHostToDevice));

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(length/(double)block_dim);
	const int shared_memory = block_dim*sizeof(double);

	DEVICE_TIC(0);
	if(config.kernel_version == CH8__PREFIX_SUM_KOGGE_STONE){
		kogge_stone_scan_by_block<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, length, NULL);
	}else if(config.kernel_version == CH8__PREFIX_SUM_BRENT_KUNG){
		brent_kung_scan_by_block<<<grid_dim, block_dim, shared_memory>>>(d_input, d_output, length, NULL);
	}else{
		printf("\nINVALID KERNEL VERSION\n");
	}

	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(double), cudaMemcpyDeviceToHost));


	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));

}

void ch8__prefix_sum_host(double *input, double *output, const int length){
	HOST_TIC(0);
	double acc = input[0];
	output[0] = acc;
	for(int i = 1; i < length; i++){
		acc += input[i];
		output[i] = acc;
	}
	HOST_TOC(0)
}

void ch8__prefix_sum(env_e env, kernel_config_t config){
	double *input, *output;

	input = (double *)malloc(CH8__ARRAY_LENGTH*sizeof(double));
	output = (double *)calloc(CH8__ARRAY_LENGTH, sizeof(double));

	nvixnu__populate_array_from_file(CH8__FILEPATH, "%lf,", CH8__ARRAY_LENGTH, sizeof(double), input);


	if(env == Host){
		ch8__prefix_sum_host(input, output, CH8__ARRAY_LENGTH);
	}else{
		ch8__prefix_sum_device(input, output, CH8__ARRAY_LENGTH, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(output + (CH8__ARRAY_LENGTH - PRINT_LENGTH), sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);


	free(input);
	free(output);

	return;
}
