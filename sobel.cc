#include <iostream>
#include <cmath>
#include "sobel.h"

float* 
sobel_filter() {
  int count = 9;
  float* filter = new float[count]; 
  filter[0] = 2;
  filter[1] = 2;
  filter[2] = 3;

  filter[3] = 2;
  filter[4] = 1;
  filter[5] = 3;

  filter[6] = 3;
  filter[7] = 3;
  filter[8] = 3;

  return filter;
}

int 
apply_filter_on_element(float filter[], int* original_image, int index, 
    int width) {
  float element = original_image[index - (width + 1)] * filter[0] +
    original_image[index - width] * filter[1] + 
    original_image[index - (width - 1)] * filter[2] + 
    original_image[index - 1] * filter[3] + 
    original_image[index] * filter[4] + 
    original_image[index + 1] * filter[5] + 
    original_image[index + (width - 1)] * filter[6] + 
    original_image[index + width] * filter[7] + 
    original_image[index + (width + 1)] * filter[8];

  if (element < 30)
    element = 0;
  return (int)(round(element));
}

// THIS IS THE ACTUAL WORK
void 
filter_on_pic(int* original_image, int* filtered_image, 
    float* filter, int height, int width) {
  int surrounded_width = width + 2;
  int surrounded_height = height + 2;
  long image_length = surrounded_height * surrounded_width;

  for (long index = surrounded_width + 1; 
      index < image_length - (surrounded_width + 1); index++) {
    long original_index = (index - (surrounded_width + 1)) - 
      (index / surrounded_width - 1) * 2;
    if ((index % surrounded_width) != 0 && 
        (index % surrounded_width) != (surrounded_width - 1))
      filtered_image[original_index] = apply_filter_on_element(filter, 
          original_image, index, surrounded_width);
  }
}
