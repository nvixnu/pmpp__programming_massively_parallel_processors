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

#define CH10__CSR_DATA_FILEPATH ""
#define CH10__CSR_COL_INDEX_FILEPATH ""
#define CH10__CSR_ROW_PTR_FILEPATH ""

#define CH10__VECTOR_FILEPATH ""

#define CH10__INPUT_NON_ZERO_LENGTH 10
#define CH10__INPUT_ROWS 10
#define CH10__INPUT_COLS 10
#define CH10__INPUT_LARGEST_NONZERO_ROW_WIDTH 10

#define CH10__SPMV_CSR "SPMV_CSR"
#define CH10__SPMV_ELL "SPMV_ELL"

/**
 * Performs the sparse matrix storage and manipulation on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch10__spmv(env_e env, kernel_config_t config);


#endif /* CHAPTER_10_H_ */
