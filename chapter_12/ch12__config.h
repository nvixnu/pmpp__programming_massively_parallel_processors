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

#define CH12__DEST_FILEPATH "../datasets/sparse/nz80666_3000x3000/nz80666_3000x3000_col_idx.txt"
#define CH12__DEST_LENGTH 80666

#define CH12__EDGES_FILEPATH "../datasets/sparse/nz80666_3000x3000/nz80666_3000x3000_row_ptr.txt"
#define CH12__EDGES_LENGTH 3001

#define CH12__MAX_FRONTIER_LENGTH 50

#define CH12__BLOCK_LEVEL_QUEUE "BLOCK_LEVEL_QUEUE"

/**
 * Performs the Breadth-first search on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch12__bfs(env_e env, kernel_config_t config);


#endif /* CHAPTER_12_H_ */
