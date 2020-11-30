/*
 * chapter_2.h
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
 * @param env Enum with the values Host = 0, Device = 1
 */
void ch2__vec_add(config_t config);



#endif /* CHAPTER_4_H_ */
