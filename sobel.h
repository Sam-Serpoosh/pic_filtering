#include <iostream>
#include <cmath>
#include <sys/time.h>

#define FILTER_SIZE 3

const int X_FILTER[FILTER_SIZE][FILTER_SIZE] =  
  {{1, 0, -1}, {2, 0, -2}, {1,  0, -1}};
const int Y_FILTER[FILTER_SIZE][FILTER_SIZE] = 
  {{1, 2, 1}, {0,  0,  0}, {-1, -2, -1}};

void 
filter_on_pic(int* original_image, int* filtered_image, 
    int height, int width);

void 
execute_filter_on_pic_and_time_it(int* original_image, 
    int* filtered_image, int height, int width);
