/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 11
 * Presents the merge sort algorithm and tiling with dynamic input data identification.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_11_H_
#define CHAPTER_11_H_

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info

#define CH11__A_FILEPATH CREDIT_CARD_DATASET_PATH
#define CH11__A_LENGTH CREDIT_CARD_DATASET_LENGTH

#define CH11__B_FILEPATH CREDIT_CARD_DATASET_PATH
#define CH11__B_LENGTH CREDIT_CARD_DATASET_LENGTH

#define CH11__C_LENGTH (CH11__A_LENGTH+CH11__B_LENGTH)

#define CH11__BASIC_MERGE_SORT "BASIC_MERGE_SORT"
#define CH11__TILED_MERGE_SORT "TILED_MERGE_SORT"
#define CH11__CIRCULAR_BUFFER_MERGE_SORT "CIRCULAR_BUFFER_MERGE_SORT"


/**
 * Performs the merge sort on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch11__merge_sort(env_e env, kernel_config_t config);


#endif /* CHAPTER_11_H_ */
