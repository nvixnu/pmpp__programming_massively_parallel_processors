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

#define CH7__2D_FILEPATH CREDIT_CARD_DATASET_PATH
#define CH7__2D_MASK_WIDTH 3
#define CH7__2D_ARRAY_WIDTH 2500
#define CH7__2D_ARRAY_HEIGHT 2500
#define CH7__2D_ARRAY_LENGTH CH7__2D_ARRAY_WIDTH*CH7__2D_ARRAY_HEIGHT

const double H_MASK[CH7__2D_MASK_WIDTH * CH7__2D_MASK_WIDTH] = {1, 0, -1, 1, 0, -1, 1, 0, -1};

/**
 * Performs the 1D convolution operation on host and device
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch7__1d_convolution(env_e env, kernel_config_t config);

/**
 * Performs the 2D convolution operation on host and device
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch7__2d_convolution(env_e env, kernel_config_t config);


#endif /* CHAPTER_7_H_ */
