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
#define CH8__ARRAY_LENGTH 1000000//CREDIT_CARD_DATASET_LENGTH

#define CH8__PREFIX_SUM_KOGGE_STONE "KOGGE_STONE"
#define CH8__PREFIX_SUM_BRENT_KUNG "BRENT_KUNG"
#define CH8__PREFIX_SUM_3_PHASE_KOGGE_STONE "3_PHASE_KOGGE_STONE"
#define CH8__HIERARCHICAL_PREFIX_SUM_KOGGE_STONE "HIERARCHICAL_KOGGE_STONE"
#define CH8__HIERARCHICAL_PREFIX_SUM_BRENT_KUNG "HIERARCHICAL_BRENT_KUNG"
#define CH8__HIERARCHICAL_PREFIX_SUM_3_PHASE_KOGGE_STONE "3_PHASE_KOGGE_STONE"
#define CH8__SINGLE_PASS_PREFIX_SUM_KOGGE_STONE "SINGLE_PASS_KOGGE_STONE"
#define CH8__SINGLE_PASS_PREFIX_SUM_3_PHASE_KOGGE_STONE "SINGLE_PASS_3_PHASE_KOGGE_STONE"

/**
 * Performs the prefix sum by sections on host and device.
 * Each block is equivalent to an array section, so the scan is performed in each section of blockDim.x elements (up to maxThreadsPerBlock)
 * The config.kernel_version param values are:
 * 	CH8__PREFIX_SUM_KOGGE_STONE: Kogge-Stone inclusive section scan (blockDim.x bound)
 * 	CH8__PREFIX_SUM_BRENT_KUNG: Brent-Kung inclusive section scan (blockDim.x bound)
 * 	CH8__PREFIX_SUM_KOGGE_STONE_3_PHASE: Three phase Kogge-Stone inclusive section scan (sharedMemPerBlock bound)
 * 	CH8__PREFIX_SUM_BRENT_KUNG_3_PHASE: Three phase Brent-Kung inclusive section scan (sharedMemPerBlock bound)
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 * @param host_section_length Specify the host section lenght in order to emulate the device constraints (Limited amount of threads and shared memory)
 */
void ch8__partial_prefix_sum(env_e env, kernel_config_t config, const int host_section_length);

/**
 * Performs the prefix sum on host and device.
 * The partial preffix sum method are called by the hierarchical and single pass scan algorithms in order  to perform the prefix sum in the entire input array.
 * The array must have up to maxGridSize[0]*maxThreadsPerBlock. Ex: 65,536*1024 = 67,108,864
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch8__full_prefix_sum(env_e env, kernel_config_t config);

#endif /* CHAPTER_8_H_ */
