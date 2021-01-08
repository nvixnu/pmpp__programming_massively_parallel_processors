/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 9
 * Presents the parallel histogram with the privatization and aggegation techniques.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_9_H_
#define CHAPTER_9_H_

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info

#define CH9__FILEPATH LOREM_IPSUM_DATASET_PATH
#define CH9__INPUT_ARRAY_LENGTH LOREM_IPSUM_DATASET_BYTES
#define CH9__OUTPUT_ARRAY_LENGTH 5

#define CH9__HISTOGRAM_WITH_BLOCK_PARTITIONING "HISTOGRAM_WITH_BLOCK_PARTITIONING"
#define CH9__HISTOGRAM_WITH_INTERLEAVED_PARTITIONING "HISTOGRAM_WITH_INTERLEAVED_PARTITIONING"
#define CH9__HISTOGRAM_PRIVATIZED "HISTOGRAM_PRIVATIZED"
#define CH9__HISTOGRAM_AGGREGATED "HISTOGRAM_AGGREGATED"

/**
 * Performs the host and device parallel histogram
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch9__parallel_histogram(env_e env, kernel_config_t config);


#endif /* CHAPTER_9_H_ */
