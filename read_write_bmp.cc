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
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>
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

int main(int argc, char* argv[])
{
	header_type header;
	information_type information;
	string imageFileName, newImageFileName;
	unsigned char tempData[3];
	int row, col, row_bytes, padding;
	vector <vector <int> > data, newData;

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

	// extract image data, initialize vectors
	// matrix 'data' contains the RED values from the image
	// 		Note: in a grey-scale image, the RED, GREEN, and BLUE values are identical
	// matrix 'newdata' is a zeroed-out matrix of the same size as 'data'
	//		Note: filtered/convolved data should be placed in the matrix 'newdata'
	for (row=0; row < information.height; row++) {
		data.push_back (vector <int>());
		newData.push_back (vector <int>());
		for (col=0; col < information.width; col++) {
			imageFile.read ((char *) tempData, 3 * sizeof(unsigned char));
			data[row].push_back ((int) tempData[0]);
			newData[row].push_back ((int) 0);
		}
		if (padding)
			imageFile.read ((char *) tempData, padding * sizeof(unsigned char));
	}
	cout << imageFileName << ": " << information.width << " x " << information.height << endl;


	
	//insert processing code here
	for (row=0; row < information.height; row++)
		for (col=0; col < information.width; col++)
			newData[row][col] = data[row][col];

			
			
	// write header to new image file
	newImageFile.write ((char *) &header, sizeof(header_type));
	newImageFile.write ((char *) &information, sizeof(information_type));

	// write new image data to new image file
	for (row=0; row < information.height; row++) {
		for (col=0; col < information.width; col++) {
			tempData[0] = (unsigned char) newData[row][col];
			tempData[1] = (unsigned char) newData[row][col];
			tempData[2] = (unsigned char) newData[row][col];
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
