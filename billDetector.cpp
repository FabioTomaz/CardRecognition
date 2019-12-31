//rectangle detection from: https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect.hpp>
#include <string> 

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
/*static bool noRectangleOverImage(Mat img, vector<Point> rect){
    double imgSize = img.size().width * img.size().height;
    double rectSize = fabs(contourArea(rect));
    double diff = imgSize - rectSize;
    double percent = (diff*100)/imgSize;
    cout << imgSize <<" - "<< rectSize<<": " << percent<<endl;
    if ( (imgSize - rectSize) <= 0.01){
         
        //the diference of areas is too small: consider that the border of the image was detected as a square
        return true;
    } 
    return false;
}*/

static bool noRectangleOverImage(Mat img, vector<Point> rect){
    double imgHeight = img.size().height;
    double imgWidth = img.size().width;
    if(find(rect.begin(), rect.end(), Point(0, 0)) != rect.end()) {
        //contains
        cout << rect<<endl;
        return false;
    }
    return true;
}

void fillEdgeImage(Mat edgesIn, Mat& filledEdgesOut)
{
    Mat edgesNeg = edgesIn.clone();

    floodFill(edgesNeg, Point(0,0), CV_RGB(255,255,255));
    bitwise_not(edgesNeg, edgesNeg);
    filledEdgesOut = (edgesNeg | edgesIn);

    return;
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
    //Mat imageInv;
    //bitwise_not ( image, imageInv );
    Mat blurred(image);
    medianBlur(image, blurred, 9);

    Mat sharp;
    Mat sharpening_kernel = (Mat_<double>(3, 3) << -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1);
    filter2D(blurred, sharp, CV_8U, sharpening_kernel);

    //GaussianBlur(image, blurred, Size(9, 9), 8);

    // original, downscale and upscaler was used, changed to blurr : https://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection
    
    /*namedWindow("inverted", WINDOW_NORMAL);
    resizeWindow("inverted", 800, 800);
    imshow("inverted", sharp);*/


    Mat gray0(blurred.size(), CV_8U), gray, timg, pyr;
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

        /*namedWindow("inverted", WINDOW_NORMAL);
        resizeWindow("inverted", 800, 800);
        imshow("inverted", gray0);*/


        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 3);
                
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1),1);
                //erode(gray, gray, Mat(), Point(-1,-1), 1);
                
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); //CV_RETR_EXTERNAL -> dont detect any inner squares inside the bills!

            string cstring = to_string(c);
            string name = "median blurred ";
            name+= cstring;
            name+= ", ";
            name+= to_string(l);
            namedWindow(name, WINDOW_NORMAL);
            resizeWindow(name, 800, 800);
            imshow(name, gray);

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
                    //&& noRectangleOverImage(gray, approx)
                    if( maxCosine < 0.3){
                        squares.push_back(approx);
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
    groupRectangles(rects, 1, 0.05); //we may have multiple rectangles really close, so we merge them together. at least 2 triangles must be merged and they must be really overlaped to be merged
    return rects;
}

//https://stackoverflow.com/questions/5906693/how-to-reduce-the-number-of-colors-in-an-image-with-opencv
Mat quantizeImage(const cv::Mat& inImage, int numBits)
{
    cv::Mat retImage = inImage.clone();

    uchar maskBit = 0xFF;

    // keep numBits as 1 and (8 - numBits) would be all 0 towards the right
    maskBit = maskBit << (8 - numBits);

    for(int j = 0; j < retImage.rows; j++)
        for(int i = 0; i < retImage.cols; i++)
        {
            cv::Vec3b valVec = retImage.at<cv::Vec3b>(j, i);
            valVec[0] = valVec[0] & maskBit;
            valVec[1] = valVec[1] & maskBit;
            valVec[2] = valVec[2] & maskBit;
            retImage.at<cv::Vec3b>(j, i) = valVec;
        }

        return retImage;
}


static void getDominantHSVColor(Mat image){
     //vector<Mat> channels;
        Mat image_hsv;
        cvtColor(image, image_hsv, CV_BGR2HSV);
        // Quanta Ratio
        int scale = 10;

        int hbins = 36, sbins = 25, vbins = 25;
        int histSize[] = {hbins, sbins, vbins};

        float hranges[] = { 0, 180 };
        float sranges[] = { 0, 256 };
        float vranges[] = { 0, 256 };

        const float* ranges[] = { hranges, sranges, vranges };
        MatND hist;

        int channels[] = {0, 1, 2};

        calcHist( &image_hsv, 1, channels, Mat(), // do not use mask
                hist, 3, histSize, ranges,
                true, // the histogram is uniform
                false );

        int maxVal = 0;

        int hue = 0; 
        int saturation = 0; 
        int value = 0; 

        for( int h = 0; h < hbins; h++ )
            for( int s = 0; s < sbins; s++ )
                for( int v = 0; v < vbins; v++ )
                    {
                        int binVal = hist.at<int>(h, s, v);
                        if(binVal > maxVal)
                        {
                            maxVal = binVal;

                            hue = h;
                            saturation = s;
                            value = v;
                        }
                    }

        hue = hue * scale; // angle 0 - 360
        saturation = saturation * scale; // 0 - 255
        value = value * scale; // 0 - 255

        //cout << hue <<", " <<saturation <<", " <<value<<  endl;
        Mat1b mask(image.size(), uchar(0));
        Scalar meanIntensity = mean(image);
        cout << meanIntensity << endl;
}

static Mat cropImage(Mat image, int offset){
    Rect roi;
    int offset_x = offset;
    int offset_y = offset;
    roi.x = offset_x;
    roi.y = offset_y;
    roi.width = image.size().width - (offset_x*2);
    roi.height = image.size().height - (offset_y*2);
    return image(roi);;

}

// the function draws all the squares in the image using rectangles
static vector<Mat> drawSquares( Mat& image, const vector<Rect>& squares )
{
    vector<Mat> notesImages;
    Mat image2;
    image.copyTo(image2); //avoid deep copy..
    for( size_t i = 0; i < squares.size(); i++ )
    {
        Rect rect = squares[i];
        Mat noteImage (image2, rect);
        //crop a little the image to exclude any possible background of the bill
        noteImage = cropImage(noteImage, 10);
        notesImages.push_back(noteImage);
        rectangle(image, rect, Scalar(0,255,0), 3, LINE_AA); //draw rectangle surrounding the note
    }
    namedWindow(wndname, WINDOW_NORMAL);
    resizeWindow(wndname, 800, 800);
    imshow(wndname, image);
    return notesImages;
}

// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ ){
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
    Mat image = imread(filename, IMREAD_COLOR);
    if( image.empty() ){
        cout << "Couldn't load " << filename << endl;
        return 1;
    }

    vector<Rect> rects = findSquares(image, squares);
    vector<Mat> notesImages = drawSquares(imageOriginal, rects);
    for( size_t i = 0; i < notesImages.size(); i++ ){
        Mat noteImage = notesImages[i];
        //noteImage = quantizeImage(noteImage, 4);
        string name = "note ";
        name+= to_string(i);
        cout <<name<<endl;
        getDominantHSVColor(noteImage);

        namedWindow(name, WINDOW_NORMAL);
        resizeWindow(name, 800, 800);
        imshow(name, noteImage);
    }

    int c = waitKey();
        
    return 0;
}