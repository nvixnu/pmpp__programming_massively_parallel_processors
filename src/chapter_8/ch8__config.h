/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 8
 * Presents the prefix sum algorithm by the Kogge-Stone and Brent-King designs
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 8/12/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_8_H_
#define CHAPTER_8_H_

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info

#define CH8__FILEPATH CREDIT_CARD_DATASET_PATH
#define CH8__ARRAY_LENGTH 8//CREDIT_CARD_DATASET_LENGTH

#define CH8__PREFIX_SUM_KOGGE_STONE "KOGGE_STONE"
#define CH8__PREFIX_SUM_BRENT_KUNG "BRENT_KUNG"
#define CH8__PREFIX_SUM_KOGGE_STONE_3_PHASE "KOGGE_STONE_3_PHASE"
#define CH8__PREFIX_SUM_BRENT_KUNG_3_PHASE "BRENT_KUNG_3_PHASE"

/**
 * Performs the prefix sum by sections on host and device. Each block is equivalent to an array section.
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch8__partial_prefix_sum(env_e env, kernel_config_t config);

/**
 * Performs the prefix sum on host and device.
 * The partial preffix sum method are called by the hierarchical and single pass scan algorithms in order  to perform the prefix sum in the entire input array.
 * The array must have up to (maxGridSize[0])*(2*maxThreadsPerBlock). Ex: 65,536*2*1024 = 134,217,728
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch8__full_prefix_sum(env_e env, kernel_config_t config);


#endif /* CHAPTER_8_H_ */
