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
#include "chapter_3.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "nvixnu__error_utils.h"

using namespace cv;

__global__
void color_to_grayscale_kernel(uchar *input, uchar *output, const int width, const int height){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if(col < width && row < height){
    	int gray_offset = row*width + col;
    	int rgb_offset = gray_offset * 3;
    	output[gray_offset] = 0.07*input[rgb_offset + 2] + 0.71*input[rgb_offset + 1] + 0.21*input[rgb_offset + 0];
	}
}

void ch3__color_to_grayscale_device(uchar *h_input, uchar *h_output, const int width, const int height){
	uchar *d_input, *d_output;
	const int length = width*height;

	CCE(cudaMalloc(&d_input, 3*length*sizeof(uchar)));
	CCE(cudaMalloc(&d_output, length*sizeof(uchar)));

	CCE(cudaMemcpy(d_input, h_input, 3*length*sizeof(uchar), cudaMemcpyHostToDevice));

	dim3 block_dim(32, 32, 1);
	dim3 grid_dim(ceil(width/32.0), ceil(height/32.0), 1);

	color_to_grayscale_kernel<<<grid_dim, block_dim>>>(d_input, d_output, width, height);
	CCLE();

	CCE(cudaMemcpy(h_output, d_output, length*sizeof(uchar), cudaMemcpyDeviceToHost));

	CCE(cudaFree(d_input));
	CCE(cudaFree(d_output));
}

void ch3__color_to_grayscale_host(uchar *input, uchar *output, const int width, const int height){
	for(int row = 0; row < height; row++){
	    for(int col = 0; col < width; col++){
	    	int gray_offset = row*width + col;
	    	int rgb_offset = gray_offset * 3;
	    	output[gray_offset] = 0.07*input[rgb_offset + 2] + 0.71*input[rgb_offset + 1] + 0.21*input[rgb_offset + 0];
	    }
	}
}

void ch3__color_to_grayscale(config_t config){
	// reads the image file
	Mat src = imread(INPUT_FILE, IMREAD_COLOR);
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
	if(config.env == Host){
		ch3__color_to_grayscale_host(input, output, src.cols, src.rows);
		output_filename = OUTPUT_HOST_FILE;
	}else{
		ch3__color_to_grayscale_device(input, output, src.cols, src.rows);
		output_filename = OUTPUT_DEVICE_FILE;
	}

	//Copy the output pixel array to a destination Mat opject
	Mat dst(src.rows, src.cols, CV_8UC1, output);

	// Save the grayscale image to the appropriate file
	imwrite(output_filename, dst);


	return;
}
