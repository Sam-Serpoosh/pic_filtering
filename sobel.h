#include <iostream>
#include <cmath>

using namespace std;

float* get_filter() {
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
apply_filter_on_element(float filter[], int* original, int index, 
    int width) {
  float element = original[index - (width + 1)] * filter[0] +
    original[index - width] * filter[1] + 
    original[index - (width - 1)] * filter[2] + 
    original[index - 1] * filter[3] + 
    original[index] * filter[4] + 
    original[index + 1] * filter[5] + 
    original[index + (width - 1)] * filter[6] + 
    original[index + width] * filter[7] + 
    original[index + (width + 1)] * filter[8];

  if (element < 0)
    element = 0;
  return (int)(round(element));
}

int** one_d_to_two_d(int* data, int height, int width) {
  int** two_d = new int*[height];
  for (int row = 0; row < height; row++)
    two_d[row] = new int[width];

  for (long index = 0; index < height * width; index++) { 
    int row = index / width;
    int col = index % width;
    two_d[row][col] = data[index];
  }

  return two_d;
}

// THIS IS THE ACTUAL WORK
int**
filter_on_pic(int** image_data, int height, int width) {
  float* filter = get_filter();
  int surrounded_width = width + 2;
  int surrounded_height = height + 2;
  long image_length = surrounded_height * surrounded_width;
  int* original_one_d = new int[image_length];
  int* filtered = new int[height * width];

  for(int row = 0; row < surrounded_height; row++)
    for(int col = 0; col < surrounded_width; col++)
      original_one_d[row * surrounded_width + col] = image_data[row][col];

  for (long index = surrounded_width + 1; 
      index < image_length - (surrounded_width + 1); index++) {
    long original_index = (index - (surrounded_width + 1)) - 
      (index / surrounded_width - 1) * 2;
    if ((index % surrounded_width) != 0 && 
        (index % surrounded_width) != (surrounded_width - 1))
      filtered[original_index] = apply_filter_on_element(filter, 
          original_one_d, index, surrounded_width);
  }

  return one_d_to_two_d(filtered, height, width);
}
