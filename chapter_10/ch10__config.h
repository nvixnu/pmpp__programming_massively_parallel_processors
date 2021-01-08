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

#define CH10__INPUT_NON_ZERO_LENGTH 80666//7
#define CH10__INPUT_ROWS 3000//4
#define CH10__INPUT_COLS 3000//4
#define CH10__INPUT_LARGEST_NONZERO_ROW_WIDTH 48//3

#define CH10__CSR_EXAMPLE_NAME "nz" NUM2STR(CH10__INPUT_NON_ZERO_LENGTH) "_" NUM2STR(CH10__INPUT_ROWS) "x" NUM2STR(CH10__INPUT_COLS)
#define CH10__CSR_EXAMPLE_PATH "../datasets/sparse/" CH10__CSR_EXAMPLE_NAME "/" CH10__CSR_EXAMPLE_NAME "_"
#define CH10__CSR_DATA_FILEPATH CH10__CSR_EXAMPLE_PATH "data.txt"
#define CH10__CSR_COL_INDEX_FILEPATH CH10__CSR_EXAMPLE_PATH "col_idx.txt"
#define CH10__CSR_ROW_PTR_FILEPATH CH10__CSR_EXAMPLE_PATH "row_ptr.txt"

#define CH10__VECTOR_FILEPATH CH10__CSR_EXAMPLE_PATH "vector.txt"



#define CH10__SPMV_CSR "SPMV_CSR"
#define CH10__SPMV_ELL "SPMV_ELL"

/**
 * Performs the sparse matrix storage and manipulation on host and device
 * @param env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch10__spmv(env_e env, kernel_config_t config);


#endif /* CHAPTER_10_H_ */
