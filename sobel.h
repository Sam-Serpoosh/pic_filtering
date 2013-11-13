#include <iostream>
#include <cmath>
#include <sys/time.h>

float* 
sobel_filter();

int 
apply_filter_on_element(float filter[], int* original_image, 
    int index, int width);

void 
filter_on_pic(int* original_image, int* filtered_image, 
    float* filter, int height, int width);

void 
execute_filter_on_pic_and_time_it(int* original_image, int* filtered_image, 
    float* filter, int height, int width);
