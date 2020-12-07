/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 7
 * In this chapter the convolution operation in 1D and 2D arrays is presented.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 07/12/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_7_H_
#define CHAPTER_7_H_

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info

#define CH7__1D_FILEPATH CREDIT_CARD_DATASET_PATH
#define CH7__1D_ARRAY_LENGTH CREDIT_CARD_DATASET_LENGTH
#define CH7__1D_MASK_WIDTH 3

/**
 * Performs the convolution operation (1D and 2D) on host and device
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch7__convolution_1d(env_e env, kernel_config_t config);


#endif /* CHAPTER_7_H_ */
