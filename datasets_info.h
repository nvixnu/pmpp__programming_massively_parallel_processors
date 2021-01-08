/*
 * datasets_info.h
 *
 *  Created on: 30/11/2020
 *  Author: Nvixnu
 */

#ifndef DATASETS_INFO_H_
#define DATASETS_INFO_H_

#include "utils.h"

#define PRINT_LENGTH 5

#define CREDIT_CARD_DATASET_ROWS 284807
#define CREDIT_CARD_DATASET_COLS 28
#define CREDIT_CARD_DATASET_LENGTH CREDIT_CARD_DATASET_ROWS*CREDIT_CARD_DATASET_COLS
#define CREDIT_CARD_DATASET_PATH "../datasets/credit_card_fraud/" NUM2STR(CREDIT_CARD_DATASET_ROWS) "x" NUM2STR(CREDIT_CARD_DATASET_COLS) ".csv"

#define THE_KING_JAMES_BIBLE_BYTES 4332499
#define THE_KING_JAMES_BIBLE_PATH "../datasets/texts/the-king-james-bible.txt"

#define PARALLEL_CAPACITOR_BETWEEN_PLATES_POINTS 2607745
#define PARALLEL_CAPACITOR_BETWEEN_PLATES_PATH "../datasets/parallel_capacitor_between_plates/TRI5210112_PTS2607745_p.txt"


#define IMG_FOLDER "../datasets/images/"
#define IMG_GREEN_FIELD IMG_FOLDER "green_field.jpg"
#define IMG_BEAR IMG_FOLDER "himalayan_brown_bear.jpg"

#endif /* DATASETS_INFO_H_ */
