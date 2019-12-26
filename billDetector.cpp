//rectangle detection from: https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static void help(const char* programName)
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours and contour simplification\n"
    "to find squares in a list of images (pic1-6.png)\n"
    "Returns sequence of squares detected on the image.\n"
    "Call:\n"
    "./" << programName << " [file_name (optional)]\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 2;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/**
Some contour detection detect the outer border of the image as a rectangle.. remove that
**/
static bool rectangleCoveringImage(Mat img, vector<Point> rect){
    double imgSize = img.size().width * img.size().height;
    double rectSize = fabs(contourArea(rect));
    double diff = imgSize - rectSize;
    double percent = (diff*0.1)/imgSize;
    if ( (imgSize - rectSize) > 0.005){
         cout << imgSize <<" - "<< rectSize<<": " << percent<<endl;
        //the diference of areas is too small: consider that the border of the image was detected as a square
        return false;
    } 
    return false;
}

// returns sequence of squares detected on the image.
static vector<Rect> findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    
    /*Mat greyMat, colorMat, binaryMat;
    cvtColor(image, greyMat, COLOR_BGR2GRAY);
    threshold(greyMat, binaryMat, 127,255,CV_THRESH_BINARY);
    imshow("binary", greyMat);*/

    // blur will enhance edge detection
    Mat blurred(image);
    medianBlur(image, blurred, 9);
    // original, downscale and upscaler was used, changed to blurr : https://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection

    Mat gray0(blurred.size(), CV_8U), gray, timg, pyr;

    //imshow("median blurred", blurred);

    //Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    //pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&blurred, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CHAIN_APPROX_SIMPLE);
            

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(approx)) > 1000 &&
                    isContourConvex(approx) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 && !rectangleCoveringImage(gray, approx)){
                        squares.push_back(approx);
                        //cout << fabs(contourArea(approx)) <<endl;
                    }
                }
            }
        }
    }
    vector<Rect> rects;
    for (size_t i = 0; i < squares.size(); i++){
        vector<Point> contour = squares[i];
        rects.push_back( boundingRect(contour));
    }
    //cout << rects <<endl;
    groupRectangles(rects, 1, 0.50); //at least 2 triangles must be merged
    return rects;
}


// the function draws all the squares in the image using rectangles
static void drawSquares( Mat& image, const vector<Rect>& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        Rect rect = squares[i];
        rectangle(image, rect, Scalar(0,255,0), 3, LINE_AA);
        cout << rect<<endl;

    }
    namedWindow(wndname, WINDOW_NORMAL);
    resizeWindow(wndname, 800, 800);
    imshow(wndname, image);
}

// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }
    namedWindow(wndname, WINDOW_NORMAL);
    resizeWindow(wndname, 800, 800);
    imshow(wndname, image);
}


int main(int argc, char** argv)
{
    
    string filename;
    help(argv[0]);

    if( argc > 1)
    {
     filename =  argv[1];
    }

    vector<vector<Point> > squares;

    
    Mat imageOriginal = imread(filename, IMREAD_COLOR);
    Mat image = imread(filename, IMREAD_COLOR);;
    if( image.empty() )
    {
        cout << "Couldn't load " << filename << endl;
        return 1;
    }

    vector<Rect> rects = findSquares(image, squares);
    drawSquares(imageOriginal, rects);

    int c = waitKey();
        
    return 0;
}