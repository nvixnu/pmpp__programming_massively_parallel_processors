/*
 * chapter_4.cu
 *
 *  Created on: 27/11/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <math.h>
#include "chapter_4.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__error_utils.h"
#include "nvixnu__gemm.h"


void ch4__matrix_mul_device(double *h_A, double *h_B, double *h_C, const int i_length, const int j_length, const int k_length, kernel_config_t config){
	double *d_A, *d_B, *d_C;

	CCE(cudaMalloc(&d_A, A_LENGTH*sizeof(double)));
	CCE(cudaMalloc(&d_B, B_LENGTH*sizeof(double)));
	CCE(cudaMalloc(&d_C, C_LENGTH*sizeof(double)));


	CCE(cudaMemcpy(d_A, h_A, A_LENGTH*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_B, h_B, B_LENGTH*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_C, h_C, C_LENGTH*sizeof(double), cudaMemcpyHostToDevice));

	dim3 block_dim(config.block_dim.x, config.block_dim.y, 1);
	dim3 grid_dim(ceil(K_LENGTH/(double)config.block_dim.x), ceil(I_LENGTH/(double)config.block_dim.y), 1);

	DEVICE_TIC(0);
	if(config.kernel_version == MATRIX_MUL_KERNEL_NAIVE){
		nvixnu__gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, I_LENGTH, J_LENGTH, K_LENGTH);
	}else{
		const int shared_memory_length = 2*config.block_dim.x*config.block_dim.y*sizeof(double);
		nvixnu__tiled_gemm_kernel<<<grid_dim, block_dim, shared_memory_length>>>(d_A, d_B, d_C, I_LENGTH, J_LENGTH, K_LENGTH, config.block_dim.x);
	}
	CCLE();
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_C, d_C, C_LENGTH*sizeof(double), cudaMemcpyDeviceToHost));


	CCE(cudaFree(d_A));
	CCE(cudaFree(d_B));
	CCE(cudaFree(d_C));
}

void ch4__matrix_mul_host(double *A, double *B, double *C, const int i_length, const int j_length, const int k_length){
	HOST_TIC(0);
	nvixnu__gemm_host(A, B, C, i_length, j_length, k_length);
	HOST_TOC(0);
}

void ch4__matrix_mul(env_e env, kernel_config_t config){
	double *A, *B, *C;

	A = (double*)malloc(A_LENGTH*sizeof(double));
	B = (double*)malloc(B_LENGTH*sizeof(double));
	C = (double*)calloc(C_LENGTH, sizeof(double));

	nvixnu__populate_array_from_file(MATRIX_A_PATH, "%lf,", A_LENGTH, sizeof(double), A);
	nvixnu__populate_array_from_file(MATRIX_B_PATH, "%lf,", B_LENGTH, sizeof(double), B);

	if(env == Host){
		ch4__matrix_mul_host(A, B, C, I_LENGTH, J_LENGTH, K_LENGTH);
	}else{
		ch4__matrix_mul_device(A, B, C, I_LENGTH, J_LENGTH, K_LENGTH, config);
	}


	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(C + (C_LENGTH - PRINT_LENGTH), sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(A);
	free(B);
	free(C);

	return;
}

