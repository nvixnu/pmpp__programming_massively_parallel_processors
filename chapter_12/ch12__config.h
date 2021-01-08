/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 11
 * Presents the merge sort algorithm and tiling with dynamic input data identification.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_12_H_
#define CHAPTER_12_H_

#include "../utils.h"
#include "../datasets_info.h"

#define CH12__DEST_FILEPATH "../datasets/graphs/example_destination.txt"
#define CH12__DEST_LENGTH 15

#define CH12__EDGES_FILEPATH "../datasets/graphs/example_edges.txt"
#define CH12__EDGES_LENGTH 10

#define CH12__MAX_FRONTIER_LENGTH 10

#define CH12__BLOCK_LEVEL_QUEUE "BLOCK_LEVEL_QUEUE"

/**
 * Performs the Breadth-first search on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch12__bfs(env_e env, kernel_config_t config);


#endif /* CHAPTER_12_H_ */
