/*
 * Programming Massively Parallel Processors - 3ed
 * Chapter 3
 * In this chapter the blur and color_to_grayscale functions are presented
 * The "nvixnu__" libraries used here are available at https://gist.github.com/nvixnu.
 *
 *  Created on: 30/11/2020
 *  Author: Nvixnu
 */

#include <stdio.h>
#include <time.h>
#include "chapter_3.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "nvixnu__error_utils.h"

using namespace cv;

__host__
__device__
__attribute__((always_inline))
inline void blur_unit (uchar *input, uchar *output, const int blur_size, const int width, const int height, int row, int col){
	int pix_val = 0;
	int pixels = 0;

	for(int blur_row = -blur_size; blur_row < blur_size+1; ++blur_row){
		for(int blur_col = -blur_size; blur_col < blur_size+1; ++blur_col){
			int cur_row = row + blur_row;
			int cur_col = col + blur_col;

			if(cur_row > -1 && cur_row < height && cur_col > -1 && cur_col < width){
				pix_val += input[cur_row * width + cur_col];
				pixels++;
			}
		}
	}
	output[row * width + col] = (uchar)(pix_val/pixels);
}


__global__
void blur_kernel(uchar *input, uchar *output, const int blur_size, const int width, const int height){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if(col < width && row < height){
		blur_unit(input, output, blur_size, width, height, row, col);
	}
}

void ch3__blur_device(uchar *h_input, uchar *h_output, const int blur_size, const int width, const int height, kernel_config_t config){
	uchar *d_input, *d_output;
	const int length = width*height;

	CCE(cudaMalloc(&d_input, length*sizeof(uchar)));
	CCE(cudaMalloc(&d_output, length*sizeof(uchar)));

	CCE(cudaMemcpy(d_input, h_input, length*sizeof(uchar), cudaMemcpyHostToDevice));

	dim3 block_dim(config.block_dim.x, config.block_dim.y, 1);
	dim3 grid_dim(ceil(width/(double)config.block_dim.x), ceil(height/(double)config.block_dim.y), 1);

	DEVICE_TIC(0);
	blur_kernel<<<grid_dim, block_dim>>>(d_input, d_output, blur_size, width, height);
	CCLE();
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(uchar), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));
}

void ch3__blur_host(uchar *input, uchar *output, const int blur_size, const int width, const int height){
	HOST_TIC(0);
	for(int row = 0; row < height; row++){
		for(int col = 0; col < width; col++){
			blur_unit(input, output, blur_size, width, height, row, col);
		}
	}
	HOST_TOC(0);
}

void ch3__blur(env_e env, kernel_config_t config){
	// reads the image file
	Mat src = imread(CH3__INPUT_FILE_BLUR, IMREAD_GRAYSCALE);
	// gets the total number of pixels
	int length = src.rows*src.cols; //Or  src.elemSize() * src.total()
	// Pointers to pixel arrays
	uchar *input, *output;
	const char * output_filename;


	//Check if the image can be read
	if(src.empty()){
		printf("Could not read the image!\n");
		return;
	}

	//Allocates the input and output pixel arrays
	input = (uchar *)malloc(length);
	output = (uchar *)malloc(length);

	//Copy the pixels from image to the input array
	memcpy(input, src.data, length);

	//Lauch the blur function
	if(env == Host){
		ch3__blur_host(input, output, CH3__BLUR_WIDTH, src.cols, src.rows);
		output_filename = CH3__OUTPUT_HOST_FILE_BLUR;
	}else{
		ch3__blur_device(input, output, CH3__BLUR_WIDTH, src.cols, src.rows, config);
		output_filename = CH3__OUTPUT_DEVICE_FILE_BLUR;
	}

	//Copy the output pixel array to a destination Mat opject
	Mat dst(src.rows, src.cols, CV_8UC1, output);

	// Save the grayscale image to the appropriate file
	imwrite(output_filename, dst);

	return;
}

