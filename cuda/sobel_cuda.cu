#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "sobel_cuda.h"

#define NUMBER_OF_BLOCKS 65535 
#define THREADS_PER_BLOCK 1024
#define FILTER_SIZE 9

__global__ void cu_filter_on_pic(int* original_image,int* filtered_image, 
    float* filter, int height, int width) {
  int surrounded_width = width + 2;
  int surrounded_height = height + 2;
  long image_length = surrounded_height * surrounded_width;

  int thread_id = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (thread_id < surrounded_width + 1 || 
      thread_id >= image_length - (surrounded_width + 1))
    return;

  if ((thread_id % surrounded_width) != 0 && 
      (thread_id % surrounded_width) != (surrounded_width - 1)) {
    float element = 
      original_image[thread_id - (surrounded_width + 1)] * filter[0] +
      original_image[thread_id - surrounded_width] * filter[1] + 
      original_image[thread_id - (surrounded_width - 1)] * filter[2] + 
      original_image[thread_id - 1] * filter[3] + 
      original_image[thread_id] * filter[4] + 
      original_image[thread_id + 1] * filter[5] + 
      original_image[thread_id + (surrounded_width - 1)] * filter[6] + 
      original_image[thread_id + surrounded_width] * filter[7] + 
      original_image[thread_id + (surrounded_width + 1)] * filter[8];

    if (element < 30)
      element = 0;
    element = (int)(round(element));
    long filtered_image_index = (thread_id - (surrounded_width + 1)) -
      (thread_id / surrounded_width - 1) * 2;
    filtered_image[filtered_image_index] = element;
  }
}

void filter_on_pic(int* original_image, int* filtered_image, 
    float* filter, int height, int width) {

  int* original_image_d;
  int* filtered_image_d;
  float* filter_d;
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

  result_status = cudaMalloc((void**) &filter_d, sizeof(float) * FILTER_SIZE);
  if (result_status != cudaSuccess) {
    printf("cudaMalloc - filter_d - failed\n");
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

  result_status = cudaMemcpy(filter_d, filter, 
      sizeof(float) * FILTER_SIZE, cudaMemcpyHostToDevice);
  if (result_status != cudaSuccess) {
    printf("cudaMemcpy - host-GPU - filtered_image - failed\n");
    exit(1);
  }

  dim3 dimblock(THREADS_PER_BLOCK);
  dim3 dimgrid(NUMBER_OF_BLOCKS);

  cu_filter_on_pic<<<dimgrid, dimblock>>>(original_image_d, filtered_image_d, 
      filter_d, height, width);

  result_status = cudaMemcpy(filtered_image, filtered_image_d, 
      sizeof(int) * filtered_image_size, cudaMemcpyDeviceToHost);
  if (result_status != cudaSuccess) {
    printf("cudaMemcpy - GPU-host - filtered_image_d - failed\n");
    exit(1);
  }

  cudaFree(original_image_d);
  cudaFree(filtered_image_d);
  cudaFree(filter_d);
}
