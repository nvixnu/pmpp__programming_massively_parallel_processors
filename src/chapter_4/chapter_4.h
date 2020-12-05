/*
 * chapter_4.h
 *
 *  Created on: 27/11/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_4_H_
#define CHAPTER_4_H_

#include "../utils.h"
#include "../datasets_info.h" //Credit card dataset info


#define MATRIX_A_PATH CREDIT_CARD_DATASET_PATH
#define MATRIX_B_PATH CREDIT_CARD_DATASET_PATH

#define I_LENGTH 3000
#define J_LENGTH 2000
#define K_LENGTH 1000

#define A_LENGTH I_LENGTH*J_LENGTH
#define B_LENGTH J_LENGTH*K_LENGTH
#define C_LENGTH I_LENGTH*K_LENGTH

#define MATRIX_MUL_KERNEL_NAIVE "NAIVE"
#define MATRIX_MUL_KERNEL_TILED "TILED"


/**
 * Performs the host and device matrix multiplication (MATRIX_MUL_KERNEL_NAIVE and MATRIX_MUL_KERNEL_NAIVE versions)
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch4__matrix_mul(env_e env, kernel_config_t config);


#endif /* CHAPTER_4_H_ */
