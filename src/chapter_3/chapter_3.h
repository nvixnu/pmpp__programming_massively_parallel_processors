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

#define INPUT_FILE_GRAY IMG_FOLDER "green_field.jpg"
#define OUTPUT_HOST_FILE_GRAY IMG_FOLDER "green_field_host_gray.jpg"
#define OUTPUT_DEVICE_FILE_GRAY IMG_FOLDER "green_field_device_gray.jpg"

#define INPUT_FILE_BLUR IMG_FOLDER "himalayan_brown_bear.jpg"
#define OUTPUT_HOST_FILE_BLUR IMG_FOLDER "himalayan_brown_bear_host_blur.jpg"
#define OUTPUT_DEVICE_FILE_BLUR IMG_FOLDER "himalayan_brown_bear_device_blur.jpg"

#define BLUR_SIZE 16

/**
 * Converts a colored image to grayscale
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch3__color_to_grayscale(env_e env, kernel_config_t config);

/**
 * Apply a blur mask to an image
 * @para env Environment to run on (Host or Device)
 * @param config Kernel configuration parameters such as the block dimension (Number of threads per block)
 */
void ch3__blur(env_e env, kernel_config_t config);

#endif /* CHAPTER_3_H_ */
