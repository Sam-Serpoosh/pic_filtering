using namespace std;

float* 
sobel_filter();

int 
apply_filter_on_element(float filter[], int* original_image, 
    int index, int width);

void 
filter_on_pic(int* original_image, int* filtered_image, 
    float* filter, int height, int width);
