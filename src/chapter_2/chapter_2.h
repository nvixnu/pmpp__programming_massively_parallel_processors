/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 2
 * In this chapter the vector addition and the error handlers functions are presented.
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 27/11/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_2_H_
#define CHAPTER_2_H_

#define PRINT_SIZE 7
#define FILEPATH CREDIT_CARD_DATASET_PATH
#define N CREDIT_CARD_DATASET_SIZE/2

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info

/**
 * Performs the host and device vector addition
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch2__vec_add(env_e env, kernel_config_t config);



#endif /* CHAPTER_4_H_ */
