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


#define CH4__MATRIX_A_PATH CREDIT_CARD_DATASET_PATH
#define CH4__MATRIX_B_PATH CREDIT_CARD_DATASET_PATH

#define CH4__I_LENGTH 3000
#define CH4__J_LENGTH 2000
#define CH4__K_LENGTH 1000

#define CH4__MATRIX_MUL_KERNEL_NAIVE "NAIVE"
#define CH4__MATRIX_MUL_KERNEL_TILED "TILED"


/**
 * Performs the host and device matrix multiplication
 * The config.kernel_version param values are:
 * 	MATRIX_MUL_KERNEL_NAIVE: Performs the naive (straightforward) version of matrix multiplication
 * 	MATRIX_MUL_KERNEL_TILED: Performs the tiled version of matrix multiplication
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch4__matrix_mul(env_e env, kernel_config_t config);


#endif /* CHAPTER_4_H_ */
