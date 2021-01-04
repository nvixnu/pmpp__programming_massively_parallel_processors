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
#include "ch12__config.h"
#include "nvixnu__array_utils.h"
#include "nvixnu__populate_arrays_utils.h"
#include "nvixnu__error_utils.h"

typedef struct {
	int src;
	int *edges;
	int *dest;
	int *labels;
	int *p_frontier;
	int *c_frontier;
	int edges_length;
	int dest_length;
	int max_frontier_length;
	int c_frontier_tail = 0;
	int p_frontier_tail = 0;
} bfs_t;


void insert_frontier(int val, int *array, int *tail){
	array[*tail] = val;
	(*tail)++;
}

__global__
void ch12__bfs_kernel(int *p_frontier, int *p_frontier_tail, int *c_frontier,
	int *c_frontier_tail, int *edges, int *dest, int *label, int* visited, const int block_queue_length) {
	extern __shared__ unsigned int c_frontier_s[];
	__shared__ unsigned int c_frontier_tail_s, our_c_frontier_tail;
	if(threadIdx.x == 0){
		c_frontier_tail_s = 0;
	}
	__syncthreads();
	const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < *p_frontier_tail){
		const unsigned int my_vertex = p_frontier[tid];
		for(unsigned int i = edges[my_vertex]; i < edges[my_vertex + 1]; ++i) {
			const unsigned int was_visited = atomicExch(&(visited[dest[i]]), 1);
			if(!was_visited) {
				label[dest[i]] = label[my_vertex] + 1;
				const unsigned int my_tail = atomicAdd(&c_frontier_tail_s, 1);
				if(my_tail < block_queue_length) {
					c_frontier_s[my_tail] = dest[i];
				} else { // IE full, add it to the global queve directly
					c_frontier_tail_s = block_queue_length;
					const unsigned int my_global_tail = atomicAdd(c_frontier_tail, 1);
					c_frontier[my_global_tail] = dest[i];
				}
			}
		}
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
	}
	__syncthreads();
	for (unsigned int i = threadIdx.x; i < c_frontier_tail_s; i += blockDim.x) {
		c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
	}
}


__global__
void ch12__frontier_exchange(int *p_frontier_tail, int *c_frontier_tail){
	*p_frontier_tail = *c_frontier_tail;
	*c_frontier_tail = 0;
}


void ch12__bfs_device(bfs_t h_bfs, kernel_config_t config){
	// copy edges, dest, and label to device global memory
	// allocate c_frontier_tail_d, p_frontier_tail_d in device global memory

	int *d_edges, *d_labels, *d_dest, *d_visited, *d_c_frontier, *d_p_frontier, *d_c_frontier_tail, *d_p_frontier_tail;
	int *p_frontier_tail;

	p_frontier_tail = (int *)malloc(sizeof(int));

	CCE(cudaMalloc(&d_edges, h_bfs.edges_length*sizeof(int)));
	CCE(cudaMalloc(&d_labels, h_bfs.dest_length*sizeof(int)));
	CCE(cudaMalloc(&d_dest, h_bfs.dest_length*sizeof(int)));
	CCE(cudaMalloc(&d_visited, h_bfs.dest_length*sizeof(int)));
	CCE(cudaMalloc(&d_c_frontier, h_bfs.dest_length*sizeof(int)));
	CCE(cudaMalloc(&d_p_frontier, h_bfs.dest_length*sizeof(int)));
	CCE(cudaMalloc(&d_c_frontier_tail, sizeof(int)));
	CCE(cudaMalloc(&d_p_frontier_tail, sizeof(int)));


	CCE(cudaMemcpy(d_edges, h_bfs.edges, h_bfs.edges_length*sizeof(int), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_dest, h_bfs.dest, h_bfs.dest_length*sizeof(int), cudaMemcpyHostToDevice));
	CCE(cudaMemcpy(d_labels, h_bfs.labels, h_bfs.dest_length*sizeof(int), cudaMemcpyHostToDevice));


	CCE(cudaMemset(d_c_frontier_tail, 0, sizeof(int)));
	CCE(cudaMemset(d_p_frontier, h_bfs.src, sizeof(int)));
	CCE(cudaMemset(d_p_frontier_tail, 1, sizeof(int)));


	const int block_dim = config.block_dim.x;


	DEVICE_TIC(0);
	if(!strcmp(config.kernel_version, CH12__BLOCK_LEVEL_QUEUE)){
		*p_frontier_tail = 1;
		while (*p_frontier_tail > 0) {
			int grid_dim = ceil(*p_frontier_tail/(double)block_dim);

			ch12__bfs_kernel<<<grid_dim, block_dim, 1000>>>(d_p_frontier, d_p_frontier_tail,
					d_c_frontier, d_c_frontier_tail, d_edges, d_dest, d_labels, d_visited, 100);
			CCLE();
			// use cudaMemcpy to read the *c_frontier_tail value back to host and assign
			// it to p_frontier_tail for the while-loop condition test
			CCE(cudaMemcpy(p_frontier_tail, d_c_frontier_tail, sizeof(int), cudaMemcpyDeviceToHost));

			int* temp = d_c_frontier;
			d_c_frontier = d_p_frontier;
			d_p_frontier = temp; //swap the roles
			// launch a simple kernel to set ;
			ch12__frontier_exchange<<<1, 1>>>(d_p_frontier_tail, d_c_frontier_tail);
			CCLE();
		}
	}else{
		printf("\nINVALID KERNEL VERSION\n");
		exit(1);
	}
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_bfs.labels, d_labels, h_bfs.dest_length*sizeof(int), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_edges));
	CCE(cudaFree(d_labels));
	CCE(cudaFree(d_dest));
	CCE(cudaFree(d_visited));
	CCE(cudaFree(d_c_frontier));
	CCE(cudaFree(d_p_frontier));
	CCE(cudaFree(d_c_frontier_tail));
	CCE(cudaFree(d_p_frontier_tail));

	free(p_frontier_tail);
}

void ch12__bfs_host(bfs_t bfs){

	HOST_TIC(0);
	insert_frontier(bfs.src, bfs.p_frontier, &(bfs.p_frontier_tail));

	while (bfs.p_frontier_tail > 0) {
		for (int f = 0; f < bfs.p_frontier_tail; f++) { // visit all previous frontier vertices
			int c_vertex = bfs.p_frontier[f]; // Pick up one of the previous frontier vertex
			for (int i = bfs.edges[c_vertex]; i < bfs.edges[c_vertex+1]; i++) { //for all its edges
				if (bfs.labels[bfs.dest[i]] == -1) { // The dest vertex has not been visited
					insert_frontier(bfs.dest[i], bfs.c_frontier, &(bfs.c_frontier_tail)); // overflow check omitted for brevity
					bfs.labels[bfs.dest[i]] = bfs.labels[c_vertex] +1;
				}
			}
		}
		int* temp = bfs.c_frontier;
		bfs.c_frontier = bfs.p_frontier;
		bfs.p_frontier = temp; //swap previous and current
		bfs.p_frontier_tail = bfs.c_frontier_tail;
		bfs.c_frontier_tail = 0;
	}
	HOST_TOC(0);


}

void ch12__bfs(env_e env, kernel_config_t config){
	bfs_t bfs;

	bfs.src = 0;
	bfs.dest_length = CH12__DEST_LENGTH;
	bfs.edges_length = CH12__EDGES_LENGTH;
	bfs.max_frontier_length = CH12__MAX_FRONTIER_LENGTH;

	bfs.c_frontier = (int *)calloc(bfs.max_frontier_length, sizeof(int));
	bfs.p_frontier = (int *)calloc(bfs.max_frontier_length, sizeof(int));

	bfs.dest = (int *)malloc(bfs.dest_length*sizeof(int));
	bfs.edges = (int *)malloc(bfs.edges_length*sizeof(int));
	bfs.labels = (int *)malloc(bfs.dest_length*sizeof(int));

	memset(bfs.labels, -1, bfs.dest_length*sizeof(int));
	bfs.labels[bfs.src] = 0;


	nvixnu__populate_array_from_file(CH12__DEST_FILEPATH, "%d,", bfs.dest_length, sizeof(int), bfs.dest);
	nvixnu__populate_array_from_file(CH12__EDGES_FILEPATH, "%d,", bfs.edges_length, sizeof(int), bfs.edges);

	if(env == Host){
		ch12__bfs_host(bfs);
	}else{
		ch12__bfs_device(bfs, config);
	}

	printf("Last %d values:\n", PRINT_LENGTH);
	//nvixnu__array_map(bfs.labels + bfs.dest_length - PRINT_LENGTH, sizeof(int), PRINT_LENGTH, nvixnu__print_item_int);
	nvixnu__array_map(bfs.labels, sizeof(int), bfs.edges_length-1, nvixnu__print_item_int);

	free(bfs.dest);
	free(bfs.edges);
	free(bfs.labels);
	free(bfs.c_frontier);
	free(bfs.p_frontier);

	return;
}
