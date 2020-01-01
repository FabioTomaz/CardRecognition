//rectangle detection from: https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect.hpp>
#include <string>
#include "money_detect.h"

#include <iostream>

using namespace cv;
using namespace std;

static void help(const char* programName)
{
    cout <<
    "\nA program that detects both coins and bills in an image and shows the total value of moneh for the boahs!\n"
    "Call:\n"
    "./" << programName << " [file_name (optional)]\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}


int main(int argc, char** argv)
{
    
    string filename;
    help(argv[0]);

    if( argc > 1)
    {
     filename =  argv[1];
    }


    
    Mat imageOriginal = imread(filename, IMREAD_COLOR);
    if( imageOriginal.empty() ){
        cout << "Couldn't load " << filename << endl;
        return 1;
    }

    //coin detection

    MoneyDetection moneyDetected = detectCoins(imageOriginal);

	namedWindow("Result", WINDOW_NORMAL);
	resizeWindow("Result", 600, 600);
	imshow("Result", moneyDetected.identifiedMoneyImage);

    //bill detection
    //todo: detetar e somar o valor nas notas

        
    return 0;
}