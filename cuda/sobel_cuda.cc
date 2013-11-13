#include "sobel_cuda.h"

using namespace std;

float* 
sobel_filter() {
  float* filter = new float[9]; 
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

void 
execute_filter_on_pic(int* original_image, int* filtered_image, 
    float* filter, int height, int width) {
  struct timeval start, finish;
  gettimeofday(&start, NULL);
  filter_on_pic(original_image, filtered_image, filter, height, width);
  gettimeofday(&finish, NULL);

  double duration = (finish.tv_usec - start.tv_usec) / 1000000.0;
  cout << "Execution Time: " << duration << endl;
}
