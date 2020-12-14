/*
 * timer.h
 *
 *  Created on: 30/11/2020
 *  Author: Nvixnu
 */

#ifndef UTILS_H_
#define UTILS_H_

#define STR_HELPER(x) #x

/**
 * Converts a number to string
 */
#define NUM2STR(x) STR_HELPER(x)

/**
 * Calculates the elapsed time from the struct timespec start and stop values
 */
#define HOST_DURATION_MS(start, stop) (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_nsec - start.tv_nsec) / 1e6

/**
 * Start the host timers
 */
#define HOST_TIC(n) \
		float duration##n; \
		struct timespec start##n, stop##n; \
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start##n);

/**
 * Stop the host timer and calculates and print the elapsed time
 */
#define HOST_TOC(n) \
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop##n); \
		duration##n = HOST_DURATION_MS(start##n, stop##n); \
		printf("\nHost elapsed time: %lf ms\n", duration##n);

/**
 * Start the device timers
 */
#define DEVICE_TIC(n) \
		float duration##n; \
		cudaEvent_t start##n, stop##n; \
		CCE(cudaEventCreate(&start##n)); \
		CCE(cudaEventCreate(&stop##n)); \
		CCE(cudaEventRecord(start##n));

/**
 * Stop the device timer and calculates and print the elapsed time
 */
#define DEVICE_TOC(n) \
		CCE(cudaEventRecord(stop##n)); \
		CCE(cudaEventSynchronize(stop##n)); \
		CCE(cudaEventElapsedTime(&duration##n, start##n, stop##n)); \
		printf("\nKernel elapsed time: %f ms\n", duration##n);


/**
 * Indicates which environment should run the function
 */
typedef enum {
	Host,
	Device
} env_e;

/**
 * Struct for holding 3d information
 */
typedef struct{
	int x;
	int y;
	int z;
} dim_t;

/**
 * Kernel configuration
 */
typedef struct{
	dim_t block_dim = {1024, 1, 1};
	const char *kernel_version;
	size_t shared_memory_length;
} kernel_config_t;

#endif /* UTILS_H_ */
