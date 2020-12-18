/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 10
 * Presents the sparse matrix storage and manipulation techniques.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 18/12/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_10_H_
#define CHAPTER_10_H_

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info

#define CH9__FILEPATH CREDIT_CARD_DATASET_PATH
#define CH9__ARRAY_LENGTH CREDIT_CARD_DATASET_LENGTH

/**
 * Performs the sparse matrix storage and manipulation on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch10__sparse_matrix(env_e env, kernel_config_t config);


#endif /* CHAPTER_10_H_ */
