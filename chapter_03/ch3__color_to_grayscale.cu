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
#include "ch3__config.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "nvixnu__error_utils.h"

using namespace cv;

__host__
__device__
__attribute__((always_inline))
inline void color_to_grayscale_unit (uchar *input, uchar *output, const int width, const int height, int row, int col){
	int gray_offset = row*width + col;
	int rgb_offset = gray_offset * 3;
	output[gray_offset] = 0.07*input[rgb_offset + 2] + 0.71*input[rgb_offset + 1] + 0.21*input[rgb_offset + 0];
}

__global__
void color_to_grayscale_kernel(uchar *input, uchar *output, const int width, const int height){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if(col < width && row < height){
		color_to_grayscale_unit(input, output, width, height, row, col);
	}
}

void ch3__color_to_grayscale_device(uchar *h_input, uchar *h_output, const int width, const int height, kernel_config_t config){
	uchar *d_input, *d_output;
	const int length = width*height;

	CCE(cudaMalloc(&d_input, 3*length*sizeof(uchar)));
	CCE(cudaMalloc(&d_output, length*sizeof(uchar)));

	CCE(cudaMemcpy(d_input, h_input, 3*length*sizeof(uchar), cudaMemcpyHostToDevice));

	dim3 block_dim(config.block_dim.x, config.block_dim.y, 1);
	dim3 grid_dim(ceil(width/(double)config.block_dim.x), ceil(height/(double)config.block_dim.y), 1);

	DEVICE_TIC(0);
	color_to_grayscale_kernel<<<grid_dim, block_dim>>>(d_input, d_output, width, height);
	CCLE();
	DEVICE_TOC(0);

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(uchar), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));
}

void ch3__color_to_grayscale_host(uchar *input, uchar *output, const int width, const int height){
	HOST_TIC(0);
	for(int row = 0; row < height; row++){
	    for(int col = 0; col < width; col++){
	    	color_to_grayscale_unit(input, output, width, height, row, col);
	    }
	}
	HOST_TOC(0);
}

void ch3__color_to_grayscale(env_e env, kernel_config_t config){
	// reads the image file
	Mat src = imread(CH3__INPUT_FILE_GRAY, IMREAD_COLOR);
	// gets the total number of pixels
	int length = src.rows*src.cols; //Or src.total()
	// Pointers to pixel arrays
	uchar *input, *output;
	const char * output_filename;


	//Check if the image can be read
	if(src.empty()){
		printf("Could not read the image!\n");
		return;
	}

	//Allocates the input and output pixel arrays
	input = (uchar *)malloc(3*length*sizeof(uchar)); //Or src.elemSize() * src.total()
	output = (uchar *)malloc(length*sizeof(uchar));

	//Copy the pixels from image to the input array
	memcpy(input, src.data, 3*length*sizeof(uchar));

	//Lauch the color_to_grayscale function
	if(env == Host){
		ch3__color_to_grayscale_host(input, output, src.cols, src.rows);
		output_filename = CH3__OUTPUT_HOST_FILE_GRAY;
	}else{
		ch3__color_to_grayscale_device(input, output, src.cols, src.rows, config);
		output_filename = CH3__OUTPUT_DEVICE_FILE_GRAY;
	}

	//Copy the output pixel array to a destination Mat opject
	Mat dst(src.rows, src.cols, CV_8UC1, output);

	// Save the grayscale image to the appropriate file
	imwrite(output_filename, dst);

	return;
}

int main(){
	printf("Chapter 03: [color_to_grayscale]\n\n");
	printf("Input: %s\n", CH3__INPUT_FILE_GRAY);	
	printf("Device output: %s\n", CH3__OUTPUT_DEVICE_FILE_GRAY);
	printf("Host output: %s\n\n", CH3__OUTPUT_HOST_FILE_GRAY);

	printf("Running on Device with 256 threads per block...");
	ch3__color_to_grayscale(Device, {.block_dim = {16,16,1}});

	printf("\nRunning on Device with 1024 threads per block...");
	ch3__color_to_grayscale(Device, {.block_dim = {32,32,1}});

	printf("\nRunning on Host...");
	ch3__color_to_grayscale(Host, {});

	return 0;
}

