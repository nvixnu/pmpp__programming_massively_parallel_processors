/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 10
 * Presents the sparse matrix storage and manipulation techniques.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <string.h>
#include "ch10__config.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"

typedef struct {
	int non_zeros;
	int rows;
	int cols;
	int largest_row_width;
} sparse_t;

typedef struct {
	double *data;
	int *col_idx;
	int *row_ptr;
} csr_t;

typedef struct {
	double *data;
	int *idx;
} ell_t;

__global__
void ch10__ell_spmv_kernel(double *m_data, int *m_col_index, const int num_cols, const int num_rows, double *v, double *y) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < num_rows) {
		float dot = 0;
		for (int col = 0; col < num_cols; col++) {
			dot += m_data[col*num_rows + row] * v[m_col_index[col*num_rows + row]];
		}
		y[row] += dot;
	}
}

__global__
void ch10__csr_spmv_kernel(double *m_data, int *m_col_index, int *m_row_ptr, const int num_rows, double *v, double *y) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < num_rows) {
		float dot = 0;
		int row_start = m_row_ptr[row];
		int row_end = m_row_ptr[row+1];
		for (int elem = row_start; elem < row_end; elem++) {
			dot += m_data[elem] * v[m_col_index[elem]];
		}
		y[row] += dot;
	}
}

void ch10__csr_spmv_device(csr_t h_csr, double *h_v, double *h_y, sparse_t dims, kernel_config_t config){
	double *d_y, *d_v;
	csr_t d_csr;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(dims.rows/(double)block_dim);

	CCE(cudaMalloc(&d_csr.data, dims.non_zeros*sizeof(double)));
	CCE(cudaMalloc(&d_csr.col_idx, dims.non_zeros*sizeof(int)));
	CCE(cudaMalloc(&d_csr.row_ptr, (dims.rows+1)*sizeof(int)));
	CCE(cudaMalloc(&d_y, dims.rows*sizeof(double)));
	CCE(cudaMalloc(&d_v, dims.cols*sizeof(double)));

	CCE(cudaMemcpy(d_csr.data, h_csr.data, dims.non_zeros*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_csr.col_idx, h_csr.col_idx, dims.non_zeros*sizeof(int), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_csr.row_ptr, h_csr.row_ptr, (dims.rows+1)*sizeof(int), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_v, h_v, dims.cols*sizeof(double), cudaMemcpyHostToDevice));

	DEVICE_TIC(0);
	ch10__csr_spmv_kernel<<<grid_dim, block_dim>>>(d_csr.data, d_csr.col_idx, d_csr.row_ptr, dims.rows, d_v, d_y);
	CCLE();
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_y, d_y, dims.rows*sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_y));
	CCE(cudaFree(d_v));
	CCE(cudaFree(d_csr.data));
	CCE(cudaFree(d_csr.col_idx));
	CCE(cudaFree(d_csr.row_ptr));
}


void ch10__ell_spmv_device(ell_t h_ell, double *h_v, double *h_y, sparse_t dims, kernel_config_t config){
	double *d_y, *d_v;
	ell_t d_ell;

	const int ell_length = dims.largest_row_width*dims.rows;

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil(dims.rows/(double)block_dim);

	CCE(cudaMalloc(&d_ell.data, ell_length*sizeof(double)));
	CCE(cudaMalloc(&d_ell.idx, ell_length*sizeof(int)));
	CCE(cudaMalloc(&d_y, dims.rows*sizeof(double)));
	CCE(cudaMalloc(&d_v, dims.cols*sizeof(double)));

	CCE(cudaMemcpy(d_ell.data, h_ell.data, ell_length*sizeof(double), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_ell.idx, h_ell.idx, ell_length*sizeof(int), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_v, h_v, dims.cols*sizeof(double), cudaMemcpyHostToDevice));


	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH10__SPMV_ELL)){
		ch10__ell_spmv_kernel<<<grid_dim, block_dim>>>(d_ell.data, d_ell.idx, dims.largest_row_width, dims.rows, d_v, d_y);
		CCLE();
	}else{
		printf("\nINVALID KERNEL VERSION\n");
		exit(1);
	}
	DEVICE_TOC(0);


	CCE(cudaMemcpy(h_y, d_y, dims.rows*sizeof(double), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_ell.data));
	CCE(cudaFree(d_y));
	CCE(cudaFree(d_v));
	CCE(cudaFree(d_ell.idx));
}


void ch10__csr_spmv_host(double *m_data, int *m_col_index, int *m_row_ptr, const int num_rows, double *v, double *y){
	HOST_TIC(0);
	for (int row = 0; row < num_rows; row++) {
		float dot = 0;
		int row_start = m_row_ptr[row];
		int row_end = m_row_ptr[row+1];
		for (int elem = row_start; elem < row_end; elem++) {
			dot += m_data[elem] * v[m_col_index[elem]];
		}
		y[row] += dot;
	}
	HOST_TOC(0)
}

void csr2ell(sparse_t dims, csr_t csr, ell_t ell){
	for (int row = 0; row < dims.rows; row++) {
		int row_start = csr.row_ptr[row];
		int row_end = csr.row_ptr[row+1];
		for(int col = 0; col < dims.largest_row_width; col++){
			int elem = row_start + col;
			int idx = col*dims.rows + row;
			if(elem < row_end){
				ell.data[idx] = csr.data[elem];
				ell.idx[idx] = csr.col_idx[elem];
			}else{
				ell.data[idx] = 0;
				ell.idx[idx] = 0;
			}
		}

	}
}

void ch10__spmv(env_e env, kernel_config_t config){
	csr_t csr;
	double *y, *v;
	sparse_t dims = {CH10__INPUT_NON_ZERO_LENGTH, CH10__INPUT_ROWS, CH10__INPUT_COLS, CH10__INPUT_LARGEST_NONZERO_ROW_WIDTH};

	csr.data = (double *)malloc(CH10__INPUT_NON_ZERO_LENGTH*sizeof(double));
	csr.col_idx = (int *)malloc(CH10__INPUT_NON_ZERO_LENGTH*sizeof(int));
	csr.row_ptr = (int *)malloc((CH10__INPUT_ROWS + 1)*sizeof(int));

	v = (double *)malloc(CH10__INPUT_COLS*sizeof(double));
	y = (double *)calloc(CH10__INPUT_ROWS, sizeof(double));

	nvixnu__populate_array_from_file(CH10__CSR_DATA_FILEPATH, "%lf", CH10__INPUT_NON_ZERO_LENGTH, sizeof(double), csr.data);
	nvixnu__populate_array_from_file(CH10__CSR_COL_INDEX_FILEPATH, "%d", CH10__INPUT_NON_ZERO_LENGTH, sizeof(int), csr.col_idx);
	nvixnu__populate_array_from_file(CH10__CSR_ROW_PTR_FILEPATH, "%d", CH10__INPUT_ROWS + 1, sizeof(int), csr.row_ptr);
	nvixnu__populate_array_from_file(CH10__VECTOR_FILEPATH, "%lf", CH10__INPUT_COLS, sizeof(double), v);

	if(env == Host){
		ch10__csr_spmv_host(csr.data, csr.col_idx, csr.row_ptr, CH10__INPUT_ROWS, v, y);
	}else{
		if(!strcmp(config.kernel_version, CH10__SPMV_CSR)){
			ch10__csr_spmv_device(csr, v, y, dims, config);
		}else{
			ell_t ell;

			const int ell_length = CH10__INPUT_LARGEST_NONZERO_ROW_WIDTH*CH10__INPUT_ROWS;

			ell.data = (double *)malloc(ell_length*sizeof(double));
			ell.idx = (int *)malloc(ell_length*sizeof(int));

			csr2ell(dims, csr, ell);

			ch10__ell_spmv_device(ell, v, y, dims, config);

			free(ell.data);
			free(ell.idx);
		}

	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(y + CH10__INPUT_ROWS - PRINT_LENGTH, sizeof(double), PRINT_LENGTH, nvixnu__print_item_double);

	free(csr.data);
	free(csr.col_idx);
	free(csr.row_ptr);
	free(y);
	free(v);

	return;
}
