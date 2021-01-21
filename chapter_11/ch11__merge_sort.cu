/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 11
 * Presents the merge sort algorithm and tiling with dynamic input data identification.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ch11__config.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"
#include "nvixnu__cuda_devices_props.h"

__device__
__host__
int minimal(int a, int b){
	return a < b ? a : b;
}

__device__
__host__
void ch11__merge_core(int *A, const int m, int *B, const int n, int *C){
	//done with A[] handle remaining B[]
	int i = 0; //index into A 3
	int j = 0; //index into B
	int k = 0; //index into C 13
	// handle the start of A[] and B[]
	while ((i < m) && (j < n)) {
		if (A[i] <= B[j]) {
			C[k++] = A[i++];
		} else {
			C[k++] = B[j++];
		}
	}
	if (i == m) {
		for (; j < n; j++) {
			C[k++] = B[j];
		}
	} else {
		for (; i <m; i++) {
			C[k++] = A[i];
		}
	}
}

__device__
__host__
void ch11__circular_buffer_merge_core(int*A, int m, int*B, int n, int*C, int A_S_start, int B_S_start, int tile_size){
	int i = 0; //virtual index into A
	int j = 0; //virtual index into B
	int k = 0; //virtual index into C
	while ((i < m) && (j < n)) {
		int i_cir= (A_S_start+ i>= tile_size) ? A_S_start+i-tile_size : A_S_start+i;
		int j_cir= (B_S_start+ j>= tile_size)? B_S_start+j-tile_size : B_S_start+j;
		if (A[i_cir] <= B[j_cir]) {
			C[k++] = A[i_cir]; i++;
		} else {
			C[k++] = B[j_cir]; j++;
		}
	}
	if (i == m) { //done with A[] handle remaining B[]
		for (; j < n; j++) {
			int j_cir = (B_S_start + j>= tile_size) ? B_S_start+j-tile_size : B_S_start+j;
			C[k++] = B[j_cir];
		}
	} else { //done with B[], handle remaining A[]
		for (; i <m; i++) {
			int i_cir = (A_S_start + i>= tile_size) ? A_S_start+i-tile_size : A_S_start+i;
			C[k++] = A[i_cir];
		}
	}
}

__device__
__host__
int ch11__co_rank(int k, int *A, int m, int *B, int n) {
	int i= minimal(k,m);
	int j = k- i;
	int i_low = 0>(k-n) ? 0 : k-n; //i_low = max(0, k-n)
	int j_low = 0>(k-m) ? 0 : k-m; //i_low = max(0, k-m)
	int delta;
	int active = 1;
	while(active){
		if (i > 0 && j < n && A[i-1] > B[j]) {
			delta = ceil((i-i_low)/2.0);
			j_low = j;
			j = j + delta;
			i = i - delta;
		} else if (j > 0 && i < m && B[j-1] >= A[i]) {
			delta = ((j - j_low +1) >> 1);
			i_low = i;
			i = i + delta;
			j = j - delta;
		}else {
			active = 0;
		}
	}
	return i;
}

__device__
__host__
int ch11__co_rank_circular(int k, int* A, int m, int* B, int n, int A_S_start, int B_S_start, int tile_size){
	int i= k<m ? k : m; //i = min (k,m)
	int j = k-i;
	int i_low = 0>(k-n) ? 0 : k-n; //i_low = max(0, k-n)
	int j_low = 0>(k-m) ? 0: k-m; //i_low = max(0,k-m)
	int delta;
	bool active = true;
	while(active){
		int i_cir = (A_S_start+i >= tile_size) ? A_S_start+i-tile_size : A_S_start+i;
		int i_m_1_cir = (A_S_start+i-1 >= tile_size)?A_S_start+i-1-tile_size: A_S_start+i-1;
		int j_cir = (B_S_start+j >= tile_size) ? B_S_start+j-tile_size : B_S_start+j;
		int j_m_1_cir = (B_S_start+i-1 >= tile_size)?B_S_start+j-1-tile_size: B_S_start+j-1;
		if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir]) {
			delta = ceil((i-i_low)/2.0);
			j_low = j;
			i = i - delta;
			j = j + delta;
		} else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
			delta = ((j - j_low +1) >> 1) ;
			i_low = i;
			i = i + delta;
			j = j - delta;
		}else{
			active = false;
		}
	}
	return i;
}

__global__
void ch11__circular_buffer_merge_kernel(int* A, int m, int* B, int n, int* C, int tile_size){
	/* shared memory allocation */
	extern __shared__ int shareAB[];
	int *A_S = &shareAB[0]; //shareA is first half of shareAB
	int *B_S = &shareAB[tile_size]; //ShareB is second half of ShareAB
	int C_curr = blockIdx.x * ceil((m+n)/(double)gridDim.x);
	// starting point of the C subarray for current block
	int C_next = minimal((blockIdx.x+1) * ceil((m+n)/(double)gridDim.x), (m+n));
	// starting point for next block
	if (threadIdx.x ==0){
		A_S[0] = ch11__co_rank(C_curr, A, m, B, n); // Make the block-level co-rank values visible to
		A_S[1] = ch11__co_rank(C_next, A, m, B, n); // other threads in the block
	}
	__syncthreads();
	int A_curr = A_S[0];
	int A_next = A_S[1];
	int B_curr = C_curr - A_curr;
	int B_next = C_next - A_next;

	__syncthreads();
	int A_S_start = 0;
	int B_S_start = 0;
	int A_S_consumed = tile_size; //in the first iteration, fill the tile_size
	int B_S_consumed = tile_size; //in the first iteration, fill the tile_size
	int counter = 0;
	int C_length = C_next - C_curr;
	int A_length = A_next - A_curr;
	int B_length = B_next - B_curr;
	int total_iteration = ceil((C_length)/(double)tile_size);
	int C_completed = 0;
	int A_consumed = 0;
	//iteration counter
	//total iteration
	int B_consumed = 0;
	while(counter < total_iteration){
		/* loading A_S_consumed elements into A_S */
		for(int i=0; i<A_S_consumed; i+=blockDim.x){
			if( i + threadIdx.x < A_length - A_consumed && i + threadIdx.x < A_S_consumed){
				A_S[(A_S_start + i + threadIdx.x)%tile_size] = A[A_curr + A_consumed + i + threadIdx.x ];
			}
		}
		/* loading B_S_consumed elements into B_S */
		for(int i=0; i<B_S_consumed; i+=blockDim.x){
			if(i + threadIdx.x < B_length - B_consumed && i + threadIdx.x < B_S_consumed){
				B_S[(B_S_start + i + threadIdx.x)%tile_size] = B[B_curr + B_consumed + i + threadIdx.x];
			}
		}

		int c_curr = threadIdx.x * (tile_size/blockDim.x);
		int c_next = (threadIdx.x+1) * (tile_size/blockDim.x);
		c_curr = (c_curr <= C_length-C_completed) ? c_curr : C_length-C_completed;
		c_next = (c_next <= C_length-C_completed) ? c_next : C_length-C_completed;
		/* find co-rank for c_curr and c_next */
		int a_curr = ch11__co_rank_circular(c_curr,
				A_S, minimal(tile_size, A_length-A_consumed),
				B_S, minimal(tile_size, B_length-B_consumed),
				A_S_start, B_S_start, tile_size);
		int b_curr = c_curr -a_curr;
		int a_next = ch11__co_rank_circular(c_next,
				A_S, minimal(tile_size, A_length-A_consumed),
				B_S, minimal(tile_size, B_length-B_consumed),
				A_S_start, B_S_start, tile_size);
		int b_next = c_next - a_next;
		/* All threads call the circular-buffer version of the sequential merge function */
		ch11__circular_buffer_merge_core( A_S, a_next-a_curr,
				B_S, b_next-b_curr, C+C_curr+C_completed+c_curr,
				A_S_start+a_curr, B_S_start+b_curr, tile_size);
		/* Figure out the work has been done */
		counter++;
		A_S_consumed = ch11__co_rank_circular(minimal(tile_size,C_length-C_completed),
				A_S, minimal(tile_size, A_length-A_consumed),
				B_S, minimal(tile_size, B_length-B_consumed),
				A_S_start, B_S_start, tile_size);
		B_S_consumed = minimal(tile_size, C_length-C_completed) -A_S_consumed;
		A_consumed+= A_S_consumed;
		C_completed += minimal(tile_size, C_length-C_completed);
		B_consumed = C_completed -A_consumed;
		A_S_start = A_S_start + A_S_consumed;

		if (A_S_start >= tile_size){
			A_S_start = A_S_start -tile_size;
		}
		B_S_start = B_S_start + B_S_consumed;

		if (B_S_start >= tile_size){
			B_S_start = B_S_start -tile_size;
		}
		__syncthreads();
	}
}


__global__
void ch11__tiled_merge_kernel(int* A, int m, int* B, int n, int* C, int tile_size){
	/* shared memory allocation */
	extern __shared__ int shareAB[];
	int *A_S = &shareAB[0]; //shareA is first half of shareAB
	int *B_S = &shareAB[tile_size]; //ShareB is second half of ShareAB

	int C_curr = blockIdx.x * ceil((m+n)/(double)gridDim.x) ;
	// starting point of the C subarray for current block
	int C_next = minimal((int)(blockIdx.x+1)* ceil((m+n)/(double)gridDim.x), (m+n));
	// starting point for next block
	if (threadIdx.x == 0){
		A_S[0] = ch11__co_rank(C_curr, A, m, B, n); // Make the block-level co-rank values visible to
		A_S[1] = ch11__co_rank(C_next, A, m, B, n); // other threads in the block
	}
	__syncthreads();

	int A_curr = A_S[0];
	int A_next = A_S[1];
	int B_curr = C_curr - A_curr;
	int B_next = C_next - A_next;
	__syncthreads();

	int counter = 0;
	int C_length = C_next - C_curr;
	int A_length = A_next - A_curr;
	int B_length = B_next - B_curr;
	int total_iteration = ceil(C_length/(double)tile_size);
	int C_completed = 0;
	int A_consumed = 0;

	//iteration counter
	//total iteration
	int B_consumed = 0;
	while(counter < total_iteration){
		/* loading tile-size A and B elements into shared memory */
		for(int i=0; i<tile_size; i+=blockDim.x){
			if( i + threadIdx.x < A_length - A_consumed){
				A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x ];
			}
		}
		for(int i=0; i<tile_size; i+=blockDim.x){
			if(i + threadIdx.x < B_length - B_consumed)
			{
				B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
			}
		}
		__syncthreads();

		int c_curr = threadIdx.x * (tile_size/blockDim.x);
		int c_next = (threadIdx.x+1) * (tile_size/blockDim.x);
		c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
		c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

		/* find co-rank for c_curr and c_next */
		int a_curr = ch11__co_rank(c_curr, A_S, minimal(tile_size, A_length-A_consumed), B_S, minimal(tile_size, B_length-B_consumed));
		int b_curr = c_curr - a_curr;
		int a_next = ch11__co_rank(c_next, A_S, minimal(tile_size, A_length-A_consumed), B_S, minimal(tile_size, B_length-B_consumed));
		int b_next = c_next - a_next;
		/* All threads call the sequential merge function */
		ch11__merge_core(A_S+a_curr, a_next-a_curr, B_S+b_curr, b_next-b_curr,C+C_curr+C_completed+c_curr);
		/* Update the A and B elements that have been consumed thus far */
		counter++;
		C_completed += tile_size;
		A_consumed += ch11__co_rank(tile_size, A_S, tile_size, B_S, tile_size);
		B_consumed = C_completed - A_consumed;
		__syncthreads();
	}
}


__global__
void ch11__basic_merge_kernel(int *A, int m, int *B, int n, int* C){
	int tid= blockIdx.x*blockDim.x + threadIdx.x;
	int num_threads = blockDim.x*gridDim.x;
	int section_size = ceil((m+n)/(double)num_threads);
	int k_curr = tid*section_size; // start index of output
	int k_next = minimal((tid+1)*section_size, m+n); // end index of output

	int i_curr= ch11__co_rank(k_curr, A, m, B, n);
	int i_next = ch11__co_rank(k_next, A, m, B, n);
	int j_curr = k_curr -i_curr;
	int j_next = k_next-i_next;
	/* All threads call the sequential merge function */
	ch11__merge_core(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr] );
}

void ch11__merge_sort_device(int *h_A, const int m, int *h_B, const int n, int *h_C, kernel_config_t config){
	int *d_A, *d_B, *d_C;

	//nvixnu__print_cuda_devices_props();

	CCE(cudaMalloc(&d_A, m*sizeof(int)));
	CCE(cudaMalloc(&d_B, n*sizeof(int)));
	CCE(cudaMalloc(&d_C, (m+n)*sizeof(int)));

	CCE(cudaMemcpy(d_A, h_A, m*sizeof(int), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_B, h_B, n*sizeof(int), cudaMemcpyHostToDevice));

	const int block_dim = config.block_dim.x;
	const int grid_dim = ceil((m+n)/(double)block_dim);
	size_t shared_memory = 2*block_dim*sizeof(int);

	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH11__BASIC_MERGE_SORT)){
		ch11__basic_merge_kernel<<<grid_dim, block_dim>>>(d_A, m, d_B, n, d_C);
	}else if(!strcmp(config.kernel_version, CH11__TILED_MERGE_SORT)){
		ch11__tiled_merge_kernel<<<grid_dim, block_dim, shared_memory>>>(d_A, m, d_B, n, d_C, block_dim);
	}else if(!strcmp(config.kernel_version, CH11__CIRCULAR_BUFFER_MERGE_SORT)){
		ch11__circular_buffer_merge_kernel<<<grid_dim, block_dim, shared_memory>>>(d_A, m, d_B, n, d_C, block_dim);
	}else{
		printf("\nINVALID KERNEL VERSION\n");
		exit(1);
	}
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_C, d_C, (m+n)*sizeof(int), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_A));
	CCE(cudaFree(d_B));
	CCE(cudaFree(d_C));
}

void ch11__merge_sort_host(int *A, const int m, int *B, const int n, int *C){
	HOST_TIC(0);
	ch11__merge_core(A, m, B, n, C);
	HOST_TOC(0)
}

void ch11__merge_sort(env_e env, kernel_config_t config){
	int *A, *B, *C;

	A = (int *)malloc(CH11__A_LENGTH*sizeof(int));
	B = (int *)malloc(CH11__B_LENGTH*sizeof(int));
	C = (int *)malloc(CH11__C_LENGTH*sizeof(int));

	nvixnu__populate_array_from_file(CH11__A_FILEPATH, "%d,", CH11__A_LENGTH, sizeof(int), A);
	nvixnu__populate_array_from_file(CH11__B_FILEPATH, "%d,", CH11__B_LENGTH, sizeof(int), B);

	if(env == Host){
		ch11__merge_sort_host(A, CH11__A_LENGTH, B, CH11__B_LENGTH, C);
	}else{
		ch11__merge_sort_device(A, CH11__A_LENGTH, B, CH11__B_LENGTH, C, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	nvixnu__array_map(C, sizeof(int), CH11__C_LENGTH, nvixnu__print_item_int);

	free(A);
	free(B);
	free(C);

	return;
}

int main(){
	printf("Chapter 11\n");
	printf("Array A length: %d\n", CH11__A_LENGTH);
	printf("Array B length: %d\n", CH11__B_LENGTH);

	printf("\n_____ merge_sort _____\n\n");

	printf("Running on Device with 256 threads per block...");
	ch11__merge_sort(Device, {.block_dim = {256, 1, 1}, .kernel_version = CH11__BASIC_MERGE_SORT});

	printf("\n_____ merge_sort_tiled _____\n\n");
	
	printf("Running on Device with 256 threads per block...");
	ch11__merge_sort(Device, {.block_dim = {256}, .kernel_version = CH11__TILED_MERGE_SORT});

	printf("\n_____ merge_sort_circular_buffer _____\n\n");

	printf("Running on Device with 256 threads per block...");
	ch11__merge_sort(Device, {.block_dim = {256, 1, 1}, .kernel_version = CH11__CIRCULAR_BUFFER_MERGE_SORT});


	printf("\n_____ merge_sort_CPU_____\n");
	ch11__merge_sort(Host, {});

	return 0;
}
