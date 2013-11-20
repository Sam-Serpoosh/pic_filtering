#include <iostream>
#include <cmath>
#include <sys/time.h>

void filter_on_pic(int*, int*, int, int);

void 
execute_filter_on_pic(int* original_image, int* filtered_image, 
  int height, int width);
