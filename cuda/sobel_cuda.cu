#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "sobel_cuda.h"

#define NUMBER_OF_BLOCKS 65535 
#define THREADS_PER_BLOCK 1024
#define FILTER_SIZE 3

__constant__ int X_FILTER[FILTER_SIZE][FILTER_SIZE] = 
  {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
__constant__ int Y_FILTER[FILTER_SIZE][FILTER_SIZE] = 
  {{1, 2, 2}, {0, 0, 0}, {-1, -2, -1}};

__global__ void cu_filter_on_pic(int* original_image,int* filtered_image, 
    int height, int width) {
  int surrounded_width = width + 2;
  int surrounded_height = height + 2;
  long image_length = surrounded_height * surrounded_width;

  int pix_horizontal_value = 0;
  int pix_vertical_value = 0;
  int thread_id = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (thread_id < surrounded_width + 1 || 
      thread_id >= image_length - (surrounded_width + 1))
    return;

  if ((thread_id % surrounded_width) != 0 && 
      (thread_id % surrounded_width) != (surrounded_width - 1)) {
    pix_horizontal_value = 
      X_FILTER[0][0] * original_image[thread_id - (width + 1)] +
      X_FILTER[0][1] * original_image[thread_id - width] + 
      X_FILTER[0][2] * original_image[thread_id - (width - 1)] + 
      X_FILTER[1][0] * original_image[thread_id - 1] + 
      X_FILTER[1][1] * original_image[thread_id] + 
      X_FILTER[1][2] * original_image[thread_id + 1] + 
      X_FILTER[2][0] * original_image[thread_id + (width - 1)] + 
      X_FILTER[2][1] * original_image[thread_id + width] + 
      X_FILTER[2][2] * original_image[thread_id + (width + 1)];
    pix_vertical_value = 
      Y_FILTER[0][0] * original_image[thread_id - (width + 1)] +
      Y_FILTER[0][1] * original_image[thread_id - width] + 
      Y_FILTER[0][2] * original_image[thread_id - (width - 1)] + 
      Y_FILTER[1][0] * original_image[thread_id - 1] + 
      Y_FILTER[1][1] * original_image[thread_id] + 
      Y_FILTER[1][2] * original_image[thread_id + 1] + 
      Y_FILTER[2][0] * original_image[thread_id + (width - 1)] + 
      Y_FILTER[2][1] * original_image[thread_id + width] + 
      Y_FILTER[2][2] * original_image[thread_id + (width + 1)];

    int pix_value = sqrtf((pix_horizontal_value * pix_horizontal_value) + 
      (pix_vertical_value * pix_vertical_value));
    if (pix_value > 255)
      pix_value = 255;
    
    // Write filtered data to filtered_image array!
    long filtered_image_index = (thread_id - (surrounded_width + 1)) -
      (thread_id / surrounded_width - 1) * 2;
    filtered_image[filtered_image_index] = pix_value;
  }
}

void filter_on_pic(int* original_image, int* filtered_image, 
    int height, int width) {
  int* original_image_d;
  int* filtered_image_d;
  int surrounded_width = width + 2;
  int surrounded_height = height + 2;
  int original_image_size = surrounded_width * surrounded_height;
  int filtered_image_size = height * width;
  cudaError_t result_status;

  result_status = cudaMalloc((void**) &original_image_d, 
      sizeof(int) * original_image_size);
  if (result_status != cudaSuccess) {
    printf("cudaMalloc - original_image_d - failed\n");
    exit(1);
  }

  result_status = cudaMalloc((void**) &filtered_image_d, 
      sizeof(int) * filtered_image_size);
  if (result_status != cudaSuccess) {
    printf("cudaMalloc - filtered_image_d - failed\n");
    exit(1);
  }

  result_status = cudaMemcpy(original_image_d, original_image, 
      sizeof(int) * original_image_size, cudaMemcpyHostToDevice);
  if (result_status != cudaSuccess) {
    printf("cudaMemcpy - host-GPU - original_image - failed\n");
    exit(1);
  }

  result_status = cudaMemcpy(filtered_image_d, filtered_image, 
      sizeof(int) * filtered_image_size, cudaMemcpyHostToDevice);
  if (result_status != cudaSuccess) {
    printf("cudaMemcpy - host-GPU - filtered_image - failed\n");
    exit(1);
  }

  dim3 dimblock(THREADS_PER_BLOCK);
  dim3 dimgrid(NUMBER_OF_BLOCKS);

  cu_filter_on_pic<<<dimgrid, dimblock>>>(original_image_d, filtered_image_d, 
      height, width);

  result_status = cudaMemcpy(filtered_image, filtered_image_d, 
      sizeof(int) * filtered_image_size, cudaMemcpyDeviceToHost);
  if (result_status != cudaSuccess) {
    printf("cudaMemcpy - GPU-host - filtered_image_d - failed\n");
    exit(1);
  }

  cudaFree(original_image_d);
  cudaFree(filtered_image_d);
}
