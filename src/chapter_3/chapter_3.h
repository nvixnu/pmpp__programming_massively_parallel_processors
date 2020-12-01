/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 3
 * In this chapter the blur and color_to_grayscale functions are presented
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 30/11/2020
 *  Author: Nvixnu
 */


#ifndef CHAPTER_3_H_
#define CHAPTER_3_H_

#include "../utils.h"


#define IMG_FOLDER "../datasets/images/"
#define INPUT_FILE IMG_FOLDER "green_field.jpg"
#define OUTPUT_HOST_FILE IMG_FOLDER "gray_field_host.jpg"
#define OUTPUT_DEVICE_FILE IMG_FOLDER "gray_field_device.jpg"

/**
 * Converts a colored image to grayscale
 * @param config Struct with the execution configuration such as environment (Host or Device) and block dimension (Number of threads per block)
 */
void ch3__color_to_grayscale(config_t config);

#endif /* CHAPTER_3_H_ */
