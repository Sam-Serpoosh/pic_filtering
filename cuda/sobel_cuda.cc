#include "sobel_cuda.h"

using namespace std;

void 
execute_filter_on_pic(int* original_image, int* filtered_image, 
    int height, int width) {
  struct timeval start, finish;
  gettimeofday(&start, NULL);
  filter_on_pic(original_image, filtered_image, height, width);
  gettimeofday(&finish, NULL);

  double duration = (finish.tv_usec - start.tv_usec) / 1000000.0;
  cout << "Execution Time: " << duration << endl;
}
