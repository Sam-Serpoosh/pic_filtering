#include "sobel.h"

using namespace std;

int 
apply_x_filter_on_element(int* original_image, int index, int width) {
  int pix_horizontal_value = X_FILTER[0][0] * original_image[index - (width + 1)] +
    X_FILTER[0][1] * original_image[index - width] + 
    X_FILTER[0][2] * original_image[index - (width - 1)] + 
    X_FILTER[1][0] * original_image[index - 1] + 
    X_FILTER[1][1] * original_image[index] + 
    X_FILTER[1][2] * original_image[index + 1] + 
    X_FILTER[2][0] * original_image[index + (width - 1)] + 
    X_FILTER[2][1] * original_image[index + width] + 
    X_FILTER[2][2] * original_image[index + (width + 1)];

  return pix_horizontal_value;
}

int apply_y_filter_on_element(int* original_image, int index, int width) {
  int pix_vertical_value = Y_FILTER[0][0] * original_image[index - (width + 1)] +
    Y_FILTER[0][1] * original_image[index - width] + 
    Y_FILTER[0][2] * original_image[index - (width - 1)] + 
    Y_FILTER[1][0] * original_image[index - 1] + 
    Y_FILTER[1][1] * original_image[index] + 
    Y_FILTER[1][2] * original_image[index + 1] + 
    Y_FILTER[2][0] * original_image[index + (width - 1)] + 
    Y_FILTER[2][1] * original_image[index + width] + 
    Y_FILTER[2][2] * original_image[index + (width + 1)];

  return pix_vertical_value;
}

int 
apply_filter_on_element(int* original_image, int index, int width) {
  int pix_horizontal_value = apply_x_filter_on_element(original_image, index, width);
  int pix_vertical_value = apply_y_filter_on_element(original_image, index, width);

  int pix_value = sqrt((pix_horizontal_value * pix_horizontal_value) + 
      (pix_vertical_value * pix_vertical_value));
  if (pix_value > 255)
    pix_value = 255;

  return pix_value;
}

int 
calculate_index_in_original_image(int index, int surrounded_width) {
  return (index - (surrounded_width + 1)) - 
    (index / surrounded_width - 1) * 2;
}

bool not_on_edge_column_of_surrounded_image(int index, int surrounded_width) {
  return (index % surrounded_width) != 0 && 
    (index % surrounded_width) != (surrounded_width - 1);
}

void 
filter_on_pic(int* original_image, int* filtered_image, 
    int height, int width) {
  int surrounded_width = width + 2;
  int surrounded_height = height + 2;
  long image_length = surrounded_height * surrounded_width;

  for (long index = surrounded_width + 1; 
      index < image_length - (surrounded_width + 1); index++) {
    long original_index = calculate_index_in_original_image(index, surrounded_width);
    if (not_on_edge_column_of_surrounded_image(index, surrounded_width))
      filtered_image[original_index] = 
        apply_filter_on_element(original_image, index, surrounded_width);
  }
}

void 
execute_filter_on_pic_and_time_it(int* original_image, 
    int* filtered_image, int height, int width) {
  struct timeval start, finish;
  gettimeofday(&start, NULL);
  filter_on_pic(original_image, filtered_image, height, width);
  gettimeofday(&finish, NULL);

  double duration = (finish.tv_usec - start.tv_usec) / 1000000.0;
  cout << "Execution Time: " << duration << endl;
}
