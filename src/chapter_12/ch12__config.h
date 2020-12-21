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
#include "../datasets_info.h" //Credit card dataset info

#define CH12__DEST_FILEPATH CREDIT_CARD_DATASET_PATH
#define CH12__DEST_LENGTH CREDIT_CARD_DATASET_LENGTH

#define CH12__EDGES_FILEPATH CREDIT_CARD_DATASET_PATH
#define CH12__EDGES_LENGTH CREDIT_CARD_DATASET_LENGTH

#define CH12__MAX_FRONTIER_LENGTH 50

/**
 * Performs the Breadth-first search on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch12__bfs(env_e env, kernel_config_t config);


#endif /* CHAPTER_12_H_ */
