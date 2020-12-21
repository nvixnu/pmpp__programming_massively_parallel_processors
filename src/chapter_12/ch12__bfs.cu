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

__global__
void BFS_Bqueve_kernel (unsigned int *p_frontier, unsigned int *p_frontier_tail, unsigned int *c_frontier,
		unsigned int *c_frontier_tail, unsigned int *edges, unsigned int *dest, unsigned int *label, unsigned int*
		visited) {
	__shared__ unsigned int c_frontier_s[BLOCK_QUEUE_SIZE];
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
				if(my_tail < BLOCK_QUEUE_SIZE) {
					c_frontier_s[my_tail] = dest[i];
				} else { // IE full, add it to the global queve directly
					c_frontier_tail_s = BLOCK_QUEUE_SIZE;
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

void BFS_host (unsigned int source, unsigned int *edges, unsigned int *dest, unsigned int *label)
{
	// allocate edges_d, dest_d, label_d, and visited_d in device global memory
	// copy edges, dest, and label to device global memory
	// allocate frontier_d, c_frontier_tail_d, p_frontier_tail_d in device global memory
	unsigned int *c_frontier_d = &frontier_d[0];
	unsigned int *p_frontier_d = &frontier_d[MAX_PRONTIER_SIZE] ;
	// launch a simple kernel to initialize the following in the device global memory
	// imitialize all visited_d elements to 0 except source to 1
	// *c_frontier_tail_d = 0;
	// p_frontier_d[0) = source;
	// *p_frontier_tail_d = 1;
	// label(source) = 0;
	p_frontier_tail = 1;
	while (p_frontier_tail > 0) {
		int num_blocks = ceil(p_frontier_tail/(double)BLOCK_SIZE);
		BFS_Bqueue_kernel<<<num_blocks, BLOCK_SIZE>>>(p_frontier_d, p_frontier_tail_d,
				c_frontier_d, c_frontier_tail_d, edges_d, dest_d, label_d, visited_d);
		// use cudaMemcpy to read the *c_frontier_tail value back to host and assign
		// it te p_frontier_tail for the while-loop condition test
		int* temp = c_frontier_d; c_frontier_d = p_frontier_d; p_frontier_d = temp; //swap the roles
		// launch a simple kernel to set *p_frontier_tail_d = *c_frontier_tail_d; *c_frontier_ctail_d = 0;
	}
}

void BPS_sequential(int source, int *edgea, int *dest, int *label)
{
	int frontier[2][MAX_FRONTIER_SIZE];
	int *e_frontier = &frontier[0];
	int c_frontier_tail = 0;
	int *p_frontier = &frontier[1];
	int p_frontier_tail = 0;
	insert_frontier(source, p_frontier, &p_frontier_tail);
	label [source] = 0;
	while (p_frontier_tail > 0) {
		for (int f = 0; f < p_frontier_tail; f++) { // visit all previous frontier vertices
			c_vertex = p_frontier[f]; // Pick up one of the previous frontier vertex
			for (int i = edges[c_vertex]; i < edges[c_vertex+1]; i++) { //for all its edges
				if (label[dest[i]] == -1) { // The dest vertex has not been visited
					insert_frontier(dest[i], c_frontier, &c_frontier_tail); // overflow check omitted for brevity
					label [dest[i]] = label[c_vertex] +1;
				}
			}
			int temp = c_frontier; c_frontier = p_frontier; p_frontier = temp; //swap previous and current
			p_frontier_tail = c_frontier_tail; c_frontier_tail = 0; //
		}
	}

	void ch12__bfs_device(int *h_input, int *h_output, const int length, kernel_config_t config){
		int *d_input, *d_output;

		CCE(cudaMalloc(&d_input, length*sizeof(int)));
		CCE(cudaMalloc(&d_output, length*sizeof(int)));

		CCE(cudaMemcpy(d_input, h_input, length*sizeof(int), cudaMemcpyHostToDevice));
		CCE(cudaMemcpy(d_output, h_output, length*sizeof(int), cudaMemcpyHostToDevice));

		DEVICE_TIC(0);
		if(!strcmp(config.kernel_version, "XXX")){

		}else{
			printf("\nINVALID KERNEL VERSION\n");
			exit(1);
		}
		DEVICE_TOC(0);

		CCE(cudaMemcpy(h_output, d_output, length*sizeof(int), cudaMemcpyDeviceToHost));

		CCE(cudaFree(d_input));
		CCE(cudaFree(d_output));
	}

	void ch12__bfs_host(int *input, int *output, const int length){
		HOST_TIC(0);
		HOST_TOC(0)
	}

	void ch12__bfs(env_e env, kernel_config_t config){
		int *dest, *edges, *labels;

		dest = (int *)malloc(CH12__DEST_LENGTH*sizeof(int));
		edges = (int *)malloc(CH12__EDGES_LENGTH*sizeof(int));
		labels = (int *)malloc(CH12__DEST_LENGTH*sizeof(int));

		nvixnu__populate_array_from_file(CH12__DEST_FILEPATH, "%d,", CH12__DEST_LENGTH, sizeof(int), dest);
		nvixnu__populate_array_from_file(CH12__EDGES_FILEPATH, "%d,", CH12__EDGES_LENGTH, sizeof(int), edges);

		if(env == Host){
			ch12__bfs_host(input, output, CH9__ARRAY_LENGTH);
		}else{
			ch12__bfs_device(input, output, CH9__ARRAY_LENGTH, config);
		}

		printf("Last %d values:\n", PRINT_LENGTH);
		nvixnu__array_map(labels + CH12__DEST_LENGTH - PRINT_LENGTH, sizeof(int), PRINT_LENGTH, nvixnu__print_item_int);

		free(dest);
		free(edges);
		free(labels);

		return;
	}
