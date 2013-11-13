#include <iostream>
#include <cmath>

void filter_on_pic(int*, int*, float*, int, int);

float* 
sobel_filter();

int 
apply_filter_on_element(float* filter, int* original_image, 
    int index, int width);

void 
execute_filter_on_pic(int* original_image, int* filtered_image, 
    float* filter, int height, int width);
