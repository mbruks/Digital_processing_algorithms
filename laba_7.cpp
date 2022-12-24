#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main(int argc, char* argv[])
{
 //Open the default video camera
 VideoCapture cap(0);
 // if not success, exit program
 if (cap.isOpened() == false)  
 {
  cout << "Cannot open the video camera" << endl;
  cin.get(); //wait for any key press
  return -1;
 } 
 double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //получаем ширину кадров видео
 double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //получаем высоту кадров видео
 Size frame_size(dWidth, dHeight);
 int frames_per_second = 10;     
 //Создаем и инициализируем объект VideoWriter
 VideoWriter oVideoWriter("MyVideo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                                                           frames_per_second, frame_size, true); 
 //Если объект VideoWriter не инициализирован успешно, тогда выходим из программы
 if (oVideoWriter.isOpened() == false) 
 {
     cout << "Cannot save the video to a file" << endl;
     cin.get(); //wait for any key press
     return -1;
 }
 string window_name = "My Camera Feed";
 namedWindow(window_name); //create a window called "My Camera Feed"
 while (true)
 {
  Mat frame;
  bool bSuccess = cap.read(frame); // read a new frame from video 
  //Прерывание цикла while, если кадры не могут быть захвачены
  if (bSuccess == false) 
  {
   cout << "Video camera is disconnected" << endl;
   cin.get(); //Wait for any key press
   break;
  }
  //write the video frame to the file
  oVideoWriter.write(frame); 
  
  if (waitKey(10) == 27)
  {
   cout << "Esc key is pressed by user. Stoppig the video" << endl;
   break;
  }
 }
 return 0;
}

