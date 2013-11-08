// readWrite_bmp.cc
//
// extracts data from .bmp file
// inserts data back into .bmp file
//
// gw

// for MSVS use Win32 Console Application / Empty Project

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

void print_content(int** content) {
  for (int row = 0; row < HEIGHT; row++) {
    for (int col = 0; col < WIDTH; col++)
      cout << content[row][col] << " ";
    cout << endl;
  }
  cout << endl << endl;
}

int** surround_with_zeros(int** data, int height, int width) {
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

int** convert_deque_to_array(deque <deque <int> > data) {
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

int** apply_filter_in_image_data(
    int** old_image, int height, int width) {
  int** surrounded = surround_with_zeros(old_image, height, width);
  ImageFilterOperator* filter = new ImageFilterOperator(surrounded, 
      height, width);
  return filter->filter_on_pic();
}

int main(int argc, char* argv[]) {
  deque <deque <int> > data;
  int** filtered_image;
  header_type header;
  information_type information;
  string imageFileName, newImageFileName;
  unsigned char tempData[3];
  int row, col, row_bytes, padding;

  // prepare files
  cout << "Original image file: ";
  cin >> imageFileName;
  ifstream imageFile;
  imageFile.open (imageFileName.c_str(), ios::binary);
  if (!imageFile) {
    cerr << "file not found" << endl;
    exit(-1);
  }
  cout << "New image file: ";
  cin >> newImageFileName;
  ofstream newImageFile;
  newImageFile.open (newImageFileName.c_str(), ios::binary);

  // read file header
  imageFile.read ((char *) &header, sizeof(header_type));
  if (header.id[0] != 'B' || header.id[1] != 'M') {
    cerr << "Does not appear to be a .bmp file.  Goodbye." << endl;
    exit(-1);
  }

  // read/compute image information
  imageFile.read ((char *) &information, sizeof(information_type));
  row_bytes = information.width * 3;
  padding = row_bytes % 4;
  if (padding)
    padding = 4 - padding;

  // extract image data, initialize deques 
  // matrix 'data' contains the RED values from the image
  // 		Note: in a grey-scale image, the RED, GREEN, and BLUE values are identical
  // matrix 'newdata' is a zeroed-out matrix of the same size as 'data'
  //		Note: filtered/convolved data should be placed in the matrix 'newdata'
  for (row=0; row < information.height; row++) {
    data.push_back (deque <int>());
    for (col=0; col < information.width; col++) {
      imageFile.read ((char *) tempData, 3 * sizeof(unsigned char));
      data[row].push_back ((int) tempData[0]);
    }
    if (padding)
      imageFile.read ((char *) tempData, padding * sizeof(unsigned char));
  }
  cout << imageFileName << ": " << information.width << " x " << information.height << endl;

  int** image_array = convert_deque_to_array(data);
  filtered_image = apply_filter_in_image_data(image_array, information.height, 
      information.width);

  // write header to new image file
  newImageFile.write ((char *) &header, sizeof(header_type));
  newImageFile.write ((char *) &information, sizeof(information_type));

  // write new image data to new image file
  for (row=0; row < information.height; row++) {
    for (col=0; col < information.width; col++) {
      tempData[0] = (unsigned char) filtered_image[row][col];
      tempData[1] = (unsigned char) filtered_image[row][col];
      tempData[2] = (unsigned char) filtered_image[row][col];
      newImageFile.write ((char *) tempData, 3 * sizeof(unsigned char));
    }
    if (padding) {
      tempData[0] = 0;
      tempData[1] = 0;
      tempData[2] = 0;
      newImageFile.write ((char *) tempData, padding * sizeof(unsigned char));
    }
  }
  cout << newImageFileName << endl;
  imageFile.close();
  newImageFile.close();

  return 0;
}
