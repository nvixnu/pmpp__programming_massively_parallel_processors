/*
 * chapter_2.h
 *
 *  Created on: 27/11/2020
 *  Author: Nvixnu
 */

#ifndef CHAPTER_2_H_
#define CHAPTER_2_H_

#define PRINT_SIZE 5
#define FILEPATH CREDIT_CARD_DATASET_PATH
#define N CREDIT_CARD_DATASET_SIZE/2

/**
 * Performs the CPU vector addition
 */
void ch2__vec_add_host(void);

/**
 * Performs the GPU vector addition
 * @param block_dim The block dimension (Number of threads in each block)
 */
void ch2__vec_add_device(const int block_dim);



#endif /* CHAPTER_4_H_ */
