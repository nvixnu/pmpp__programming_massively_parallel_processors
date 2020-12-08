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

/**
 * Performs the prefix sum on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch8__prefix_sum(env_e env, kernel_config_t config);


#endif /* CHAPTER_8_H_ */
