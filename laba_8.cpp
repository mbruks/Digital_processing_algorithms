#include <iostream> 
#include <opencv2/opencv.hpp> 
using namespace std;
using namespace cv;
//g++ show_image.cpp -o show_image `pkg-config —cflags —libs opencv4` 
int main() 
{ 
	Mat image; 
	image = imread("dino.jpg", 1); 
	 imshow("Image", image); 
	//cvtColor(image,image,COLOR_RGB2HSV);
	
	if (!image.data)
	 { 
	 	cout << "No image data \n"; 
	 	return -1;
	 } 
	 namedWindow("Image", WINDOW_AUTOSIZE);
	 imshow("Image1", image); 
	 	waitKey(0); 
	 return 0;
}
