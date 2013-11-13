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

int apply_filter_on_element(float filter[], int** original, 
    int x, int y) {
  float element = original[x - 1][y - 1] * filter[0]
    + original[x - 1][y] * filter [1]
    + original[x - 1][y + 1] * filter [2]
    + original[x][y - 1] * filter[3]
    + original[x][y] * filter[4]
    + original[x][y + 1] * filter[5]
    + original[x + 1][y - 1] * filter[6]
    + original[x + 1][y] * filter[7]
    + original[x + 1][y + 1] * filter[8];

  if (element < 0)
    element = 0;
  return (int)(round(element));
}

// THIS IS THE ACTUAL WORK
int** filter_on_pic(int** image_data, int height, int width) {
  float* filter = get_filter();
  int** filtered = new int*[height];
  for (int row = 0; row < height; row++)
    filtered[row] = new int[width];

  for(int final_x = 0; final_x < height; final_x++)
    for(int final_y = 0; final_y < width; final_y++) {
      int orig_x = final_x + 1;
      int orig_y = final_y + 1;
      filtered[final_x][final_y] = apply_filter_on_element(filter, 
          image_data, orig_x, orig_y);
    } 

  return filtered;
}
