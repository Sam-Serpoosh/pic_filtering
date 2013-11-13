#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <deque>
#include <cmath>
#include <ctime>
#include <omp.h>
#include "sobel.h"

#define HEIGHT 300
#define WIDTH 300

using namespace std;

#pragma pack(1)
typedef struct {
  char id[2]; 
  int file_size;
  int reserved;
  int offset;
}  header_type;

#pragma pack(1)
typedef struct {
  int header_size;
  int width;
  int height; 
  unsigned short int color_planes; 
  unsigned short int color_depth; 
  unsigned int compression; 
  int image_size;
  int xresolution;
  int yresolution; 
  int num_colors;
  int num_important_colors;
} information_type;

void 
print_content(int** content, int height, int width) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++)
      cout << content[row][col] << " ";
    cout << endl;
  }
  cout << endl << endl;
}

int** 
surround_with_zeros(int** data, int height, int width) {
  int surrounded_height = height + 2;
  int surrounded_width = width + 2;
  int** surrounded = new int*[surrounded_height];
  for (int row = 0; row < surrounded_height; row++)
    surrounded[row] = new int[surrounded_width];

  for (int col = 0; col < surrounded_width; col++)
    surrounded[0][col] = 0;
  for (int col = 0; col < surrounded_width; col++)
    surrounded[surrounded_height - 1][col] = 0;

  for (int row = 0; row < surrounded_height; row++)
    surrounded[row][0] = 0;

  for (int row = 1; row < surrounded_height - 1; row++)
    for (int col = 1; col < surrounded_width - 1; col++)
      surrounded[row][col] = data[row - 1][col - 1];

  return surrounded;
}

int** 
convert_deque_to_array(deque <deque <int> > data) {
  int height = data.size();
  int width = data[0].size();
  int** data_array = new int*[height];
  for (int row = 0; row < height; row++)
    data_array[row] = new int[width];

  for (int row = 0; row < height; row++)
    for (int col = 0; col < width; col++)
      data_array[row][col] = data[row][col];

  return data_array;
}

int* 
convert_two_d_to_one_d(int** image_2d, int height, int width) {
  int* one_d_image = new int[height * width];
  for(int row = 0; row < height; row++)
    for(int col = 0; col < width; col++)
      one_d_image[row * width + col] = image_2d[row][col];

  return one_d_image;
}


int** 
convert_one_d_to_two_d(int* one_d_image, int height, int width) {
  int** two_d = new int*[height];
  for (int row = 0; row < height; row++)
    two_d[row] = new int[width];

  for (long index = 0; index < height * width; index++) { 
    int row = index / width;
    int col = index % width;
    two_d[row][col] = one_d_image[index];
  }

  return two_d;
}

int** 
apply_filter_on_image(int** old_image, int height, int width) {
  int** surrounded = surround_with_zeros(old_image, height, width);
  int* one_d_surrounded = convert_two_d_to_one_d(surrounded, 
      height + 2, width + 2);
  int* filtered_image = new int[height * width];
  execute_filter_on_pic_and_time_it(one_d_surrounded, filtered_image, 
      sobel_filter(), height, width);

  return convert_one_d_to_two_d(filtered_image, height, width);
}

void 
read_image_header(ifstream& image_file, header_type& header) {
  image_file.read ((char *) &header, sizeof(header_type));
  if (header.id[0] != 'B' || header.id[1] != 'M') {
    cerr << "Does not appear to be a .bmp file.  Goodbye." << endl;
    exit(-1);
  }
}

int 
read_image_information_and_get_padding(
    ifstream& image_file, information_type& information) {
  int row_bytes, padding;
  image_file.read ((char *) &information, sizeof(information_type));
  row_bytes = information.width * 3;
  padding = row_bytes % 4;
  if (padding)
    padding = 4 - padding;

  return padding;
}

deque <deque <int> > 
read_image_data(ifstream& image_file, information_type& information, 
    unsigned char temp_data[], int padding) {
  deque <deque <int> > image_data;
  for (int row = 0; row < information.height; row++) {
    image_data.push_back(deque<int>());
    for (int col = 0; col < information.width; col++) {
      image_file.read ((char *) temp_data, 3 * sizeof(unsigned char));
      image_data[row].push_back ((int) temp_data[0]);
    }
    if (padding)
      image_file.read ((char *) temp_data, 
          padding * sizeof(unsigned char));
  }

  image_file.close();
  return image_data;
}

void 
write_image_header_and_info(ofstream& image_file, 
    header_type& header, information_type& information) {
  image_file.write ((char *) &header, sizeof(header_type));
  image_file.write ((char *) &information, sizeof(information_type));
}

void 
write_image_data(ofstream& image_file, int** filtered_image, 
    int height, int width, unsigned char temp_data[], int padding) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      temp_data[0] = (unsigned char) filtered_image[row][col];
      temp_data[1] = (unsigned char) filtered_image[row][col];
      temp_data[2] = (unsigned char) filtered_image[row][col];
      image_file.write ((char *) temp_data, 
          3 * sizeof(unsigned char));
    }

    if (padding) {
      temp_data[0] = 0;
      temp_data[1] = 0;
      temp_data[2] = 0;
      image_file.write((char *) temp_data, 
          padding * sizeof(unsigned char));
    }
  }

  image_file.close();
}

int main(int argc, char* argv[]) {
  /*int** nums;
  int height = 800; 
  int width = 800;
  nums = new int*[height];
  for (int row = 0; row < height; row++)
    nums[row] = new int[width];

  for (int row = 0; row < height; row++)
    for (int col = 0; col < width; col++)
      nums[row][col] = col + 1;

  int** result = apply_filter_on_image(nums, height, width);
  print_content(result, height, width);*/

  deque <deque <int> > original_image;
  int** filtered_image;
  header_type header;
  information_type information;
  string image_filename, new_image_file_name;
  unsigned char temp_data[3];
  int row, col, padding;
  ifstream image_file;
  ofstream new_image_file;

  cout << "Image file: ";
  cin >> image_filename;
  image_file.open (image_filename.c_str(), ios::binary);
  if (!image_file) {
    cerr << "file not found" << endl;
    exit(-1);
  }
  cout << "New image file: ";
  cin >> new_image_file_name;
  new_image_file.open (new_image_file_name.c_str(), ios::binary);

  read_image_header(image_file, header);
  padding = read_image_information_and_get_padding(image_file, 
      information);
  original_image = read_image_data(image_file, information, 
      temp_data, padding);
  int** original_image_array = convert_deque_to_array(original_image);
  filtered_image = apply_filter_on_image(original_image_array, 
      information.height, information.width);

  write_image_header_and_info(new_image_file, header, information);
  write_image_data(new_image_file, filtered_image, information.height, 
      information.width, temp_data, padding);

  cout << image_filename << ": " << information.width << 
    " x " << information.height << endl;
  cout << new_image_file_name << endl;
  return 0;
}
