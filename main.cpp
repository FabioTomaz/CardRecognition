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
    "\nA program that detects both coins and bills in an image and shows the total value of moneh for the boahs!\n"
    "Call:\n"
    "./" << programName << " [file_name (optional)]\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}


struct MoneyDetection {
    cv::Mat identifiedMoneyImage;
    int totalValue;
    int nElements;
};

/////////////////////     Helper functions  ///////////////////////////

// Converts a RGB Scalar to a HSV Scalar
// Converts a RGB Scalar to a HSV Scalar
Scalar ScalarBGR2HSV(Scalar scalar) {
    Mat hsv;
    Mat rgb(1,1, CV_8UC3, scalar);
    cvtColor(rgb, hsv, CV_BGR2HSV);
    return Scalar(hsv.data[0], hsv.data[1], hsv.data[2]);
}

// Returns mean HSV intensity from circle
Scalar getMeanCircleHSV(Mat img, Vec3f circ) {
	Mat1b mask(img.size(), uchar(0));
    circle(mask, Point(circ[0], circ[1]), circ[2], Scalar(255), CV_FILLED);
	Scalar meanIntensity = mean(img, mask);
	return ScalarBGR2HSV(meanIntensity);
}

// Converts image to square image of side target_width. Crops remains, meaning it doesn't stretch the img.
Mat getSquareImage(const cv::Mat& img, int target_width)
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}


///////////////////// Bill detection class: contains functions that recognize bills in a given image

class BillDetection 
{ 
    public: 

    //Given an image, returns a struct with: the original image with drawings surrounding the identified bills (if any), the total ammount of change detected and number of bills
    //second argument is an image to draw rectangles surrounding the identified elements. If empty, the original image is provided
    MoneyDetection detectBill(Mat image, Mat imageToDraw){
        if (imageToDraw.empty()){
            image.copyTo(imageToDraw);
        }
        int total = 0;
        
        vector<vector<Point> > squares;
        vector<Rect> rects = findSquares(image, squares);
        vector<Mat> notesImages = drawSquares(imageToDraw, rects);
        for( size_t i = 0; i < notesImages.size(); i++ ){
            Mat noteImage = notesImages[i];
            //noteImage = quantizeImage(noteImage, 4);
            string name = "note ";
            name+= to_string(i);
            cout <<name<<endl;
            Scalar hsvColor = getDominantHSVColor(noteImage);

            /*namedWindow(name, WINDOW_NORMAL);
            resizeWindow(name, 800, 800);
            imshow(name, noteImage);*/
        }
        MoneyDetection moneyDetect = {
            .identifiedMoneyImage = imageToDraw,
            .totalValue = 0,
            .nElements = 0
        };
        //waitKey();
        return moneyDetect;
    }

    private:
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


    void fillEdgeImage(Mat edgesIn, Mat& filledEdgesOut)
    {
        Mat edgesNeg = edgesIn.clone();

        floodFill(edgesNeg, Point(0,0), CV_RGB(255,255,255));
        bitwise_not(edgesNeg, edgesNeg);
        filledEdgesOut = (edgesNeg | edgesIn);

        return;
    }

    // returns sequence of squares detected on the image.
    static vector<Rect> findSquares( Mat image, vector<vector<Point> >& squares )
    {
        squares.clear();
        int thresh = 50, N = 2;
        
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

        /*namedWindow("name", WINDOW_NORMAL);
        resizeWindow("name", 800, 800);
        imshow("name", gray0);*/

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
                /*namedWindow(name, WINDOW_NORMAL);
                resizeWindow(name, 800, 800);
                imshow(name, gray);*/

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

    static Scalar getDominantHSVColor(Mat image){
        Mat1b mask(image.size(), uchar(0));
        Scalar meanIntensity = mean(image);
        cout << meanIntensity << endl;
        Scalar hsv = ScalarBGR2HSV(meanIntensity);
        cout << hsv << endl;
        return hsv;
    }

    static Mat cropImage(Mat image, int offset){
        Rect roi;
        int offset_x = offset;
        int offset_y = offset;
        roi.x = offset_x;
        roi.y = offset_y;
        roi.width = image.size().width - (offset_x*2);
        roi.height = image.size().height - (offset_y*2);
        
        return image(roi);

    }

    // the function draws all the squares in the image using rectangles
    vector<Mat> drawSquares( Mat& image, const vector<Rect>& squares )
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
        const char* wndname = "Bill Detection Demo";
        /*namedWindow(wndname, WINDOW_NORMAL);
        resizeWindow(wndname, 800, 800);
        imshow(wndname, image);*/
        return notesImages;
    }
};


class CoinDetection 
{ 
    public: 
    // Given an image, returns in a struct the original image with drawings surrounding the identified coins, total ammount and number of coins.
    //second argument is an image to draw circles surrounding the identified elements. If empty, the original image is used to draw
    MoneyDetection detectCoins(Mat img, Mat imageToDraw){
        
        Mat imgScaled, gray;
        imgScaled = getSquareImage(img, 600);

        if (imageToDraw.empty()){
            imgScaled.copyTo(imageToDraw);
        }
        cvtColor(imgScaled, gray, CV_BGR2GRAY);

        // smooth it, otherwise a lot of false circles may be detected
        GaussianBlur(gray, gray, Size(3, 3), 1.5, 1.5);

        namedWindow("Gray Blurred Image", WINDOW_NORMAL);
        resizeWindow("Gray Blurred Image", 600, 600);
        imshow("Gray Blurred Image", gray);

        Mat edges, edgesOpened;

        Canny(gray, edges, 50, 170, 3);
        
        /*
        //threshold(gray, edges, 100, 255, CV_THRESH_OTSU);
        adaptiveThreshold(edges, edges, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 1);
        //Morphology
        //Dilation
        int dilation_type = MORPH_CROSS; // dilation_type = MORPH_RECT,MORPH_CROSS,MORPH_ELLIPSE
        int dilation_size = 1;  
        Mat element = getStructuringElement(
            dilation_type,
            Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            Point(dilation_size, dilation_size)
        );
        morphologyEx(edges, edgesOpened, MORPH_ERODE, element);
        */

        namedWindow("Canny", 0);
        resizeWindow("Canny", 600, 600);
        imshow("Canny", edges);

        
        vector<Vec3f> allCircles, circles;

        // Double check for the circles - Just the edge image at this moment produces a lot of false circles - when the Hough circles function is run
        // Shortlisting good circle candidates by doing a contour analysis
        /*vector<vector<Point>> contours, contoursfil;
        vector<Vec4i> hierarchy;
        Mat contourImg2 = Mat::ones(edges.rows, edges.cols, edges.type());
        float circThresh;
        //Find all contours in the edges image
        findContours(edges.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        for (int j = 0; j < contours.size(); j++)
        {
            //Only give me contours that are closed (i.e. they have 0 or more children : hierarchy[j][2] >= 0) AND contours that dont have any parent (i.e. hierarchy[j][3] < 0 )
            if ((hierarchy[j][2] >= 0) && (hierarchy[j][3] < 0))
            {
                contoursfil.push_back(contours[j]);
            }

            circThresh = getCircularityThresh(contours[j]);
            cout <<  "Countors " << contourArea(Mat(contours[j])) << endl;
            // Doing a quick compactness/circularity test on the contours P^2/A for the circle the perfect is 12.56 .. we give some room as we mostly are extracting elliptical shapes also
            if ((circThresh > 5) && (circThresh <= 300))
            {
                contoursfil.push_back(contours[j]);
            }
            
            Point2f center;
            float radius;
            minEnclosingCircle(contours[j], center, radius);
            //cout << center << endl;
            circles.push_back(Vec3f(center.x, center.y, radius));
        }*/
        /*for (int j = 0; j < contoursfil.size(); j++)
        {
            drawContours(contourImg2, contoursfil, j, CV_RGB(255, 255, 255), 1, 8);
        }
        namedWindow("Contour Image Filtered", WINDOW_NORMAL);
        resizeWindow("Contour Image Filtered", 600, 600);
        imshow("Contour Image Filtered", contourImg2);*/
        
        // Detects cricles from canny image. param 1 and param 2 values picked based on trial and error.
        HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 2, gray.rows / 8, 200, 100, 20, 90);

        // Filter circles
        /*copy_if (
            circles.begin(), 
            circles.end(), 
            back_inserter(allCircles), 
            [](Vec3f circle){
                Scalar hsv = getMeanCircleHSV(imgScaled, circle);
                return (hsv[0] <=13 && hsv[1] >= 130 && hsv[1] <=190 && hsv[2] >= 60 && hsv[2] <=135) ||
                    (hsv[0] >= 15 && hsv[0] < 18 && hsv[1] > 50 && hsv[1] <=130 && hsv[2] > 85 && hsv[2] <=210) ||
                    (hsv[0] >= 18 && hsv[0] <=20 && hsv[1] >= 110 && hsv[1] <=160 && hsv[2] >= 95 && hsv[2] <=190);
            } 
        );*/

        //sort in descending based on radius
        struct sort_pred {
            bool operator()(const Vec3f &left, const Vec3f &right) {
                return left[2]< right[2];
            }
        };
        sort(
            circles.rbegin(), 
            circles.rend(), 
            sort_pred()
        );

        float largestRadius = circles[0][2];
        float change = 0;
        int coins = 0;
        float ratio;

        cout << circles.size() << endl;
        for (size_t i = 0; i < circles.size(); i++)
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            float radius = circles[i][2];
            ratio = ((radius*radius) / (largestRadius*largestRadius));
            
            Scalar hsv = getMeanCircleHSV(imgScaled, circles[i]);

            /*
            Average colors were picked based on trial and error. 
            There is 3 categories:
                - Pennies (color is hgsv(<= 13, 130-190, 60-110) then 5/2/1 cents)
                - Big cents (color is hgsv(15-20, 54-127, 85-207) then 10/20/50 cents)
                - Euros (color is hgsv(18-19, 110-160, 95-190) then 1/2 euros)			
            In each color category we differentiate coins based on the area of each coin compared to the 2 euro coin.
            Areas were picked based on trial and error
            */
            if (hsv[0] <=13 && hsv[1] >= 130 && hsv[1] <=190 && hsv[2] >= 60 && hsv[2] <=135)
            {
                if ((ratio >= 0.75) && (ratio<.95))
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "- 5 cents"
                    );
                    change = change + .05;
                    coins = coins + 1;
                    cout <<  i << " 5 cents - " << hsv <<endl;
                }
                else if ((ratio >= 0.65) && (ratio<.75))
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "- 2 cents"
                    );
                    change = change + .02;
                    coins = coins + 1;
                    cout <<  i << " 2 cents - " << hsv <<endl;
                }
                else if ((ratio >= 0.4) && (ratio<.65))
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "-1 cent"
                    );
                    change = change + .01;
                    coins = coins + 1;
                    cout <<  i << " 1 cent - " << hsv <<endl;
                }
            } 
            else if (hsv[0] >= 15 && hsv[0] < 18 && hsv[1] > 50 && hsv[1] <=130 && hsv[2] > 85 && hsv[2] <=210)
            {
                if (ratio >= 0.85)
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "-2 euro"
                    );
                    change = change + 2.0;
                    coins = coins + 1;
                    cout << i << "2 euro - " << hsv <<endl;
                }
                else if ((ratio >= 0.40) && (ratio<85))
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "-1 euro"
                    );
                    change = change + 1.00;
                    coins = coins + 1;
                    cout << i << " 1 euro - " << hsv <<endl;
                }
            } 
            else if (hsv[0] >= 18 && hsv[0] <=20 && hsv[1] >= 110 && hsv[1] <=160 && hsv[2] >= 95 && hsv[2] <=190)
            {
                if (ratio >= 0.90)
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "-50 cents"
                    );
                    change = change + 0.5;
                    coins = coins + 1;
                    cout << i << " 50 cents - " << hsv <<endl;
                }
                else if ((ratio >= 0.75) && (ratio<.90))
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "-20 cents"
                    );
                    change = change + .20;
                    coins = coins + 1;
                    cout << i << " 20 cents - " << hsv <<endl;
                }
                else if ((ratio >= 0.40) && (ratio<.75))
                {
                    drawResult(
                        imageToDraw, 
                        center, 
                        radius, 
                        to_string(i)+ "-10 cents"
                    );
                    change = change + .10;
                    coins = coins + 1;
                    cout << i << " 10 cents - " << hsv <<endl;
                }
            }
            
            drawResult(
                imageToDraw, 
                center, 
                radius, 
                to_string(i)+ "-?? euro"
            );
            cout <<  i << " ?? cents - " << hsv <<endl;
            

        }

        /*putText(
            imgScaled, 
            "Coins: " + to_string(change) + " // Total Money:" + to_string(change), 
            Point(imgScaled.cols / 10, imgScaled.rows - imgScaled.rows / 10), 
            FONT_HERSHEY_COMPLEX_SMALL, 
            0.7, 
            Scalar(0, 255, 255), 
            0.6, 
            CV_AA
        );*/
        MoneyDetection moneyDetect = {
            .identifiedMoneyImage = imageToDraw,
            .totalValue = change,
            .nElements = change
        };
        return moneyDetect;
    }


    private:
    // Draws a highlight in each coin and writes it's value on the side
    void drawResult(Mat img, Point center, float radius, string text) 
    {
        // draw the circle center
        circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
        // draw the circle outline
        circle(img, center, radius, Scalar(0, 0, 255), 2, 8, 0);
        rectangle(
            img, 
            Point(center.x - radius - 5, center.y - radius - 5), 
            Point(center.x + radius + 5, center.y + radius + 5), 
            CV_RGB(0, 0, 255), 
            1, 
            8, 
            0
        ); //Opened contour - draw a green rectangle arpund circle
        putText(
            img, 
            text, 
            Point(center.x - radius, center.y + radius + 15), 
            FONT_HERSHEY_COMPLEX_SMALL, 
            0.7, 
            Scalar(0, 255, 255), 
            0.4, 
            CV_AA
        );
    }

}; 


int main(int argc, char** argv)
{
    string filename;
    //help(argv[0]);
    int checkCoin = 1;
    int checkBill = 1;
    if( argc == 2)
    {
        cout << "Money recognition: detecting both coins and bills" << filename << endl;
    }
    else if( argc == 3)
    {
        if (!strcmp(argv[2],"-c")){
            cout << "Money recognition: only coins" << filename << endl;
            checkBill = 0;
        }
        else if (!strcmp(argv[2],"-b")){
            cout << "Money recognition: detecting only bills" << filename << endl;
            checkCoin = 0;
        }
    }
    else{
        cout << "Usage: ./app <image file> [-c | -b]"<<endl;
        cout << "-c will detect only coins. -b will detect only bills."<<endl;
        cout << "To detect both coins and bills insert only the image as an argument "<<endl;
        return 1;
    }
    filename =  argv[1];

    Mat imageOriginal = imread(filename, IMREAD_COLOR);
    if( imageOriginal.empty() ){
        cout << "Couldn't load " << filename << endl;
        return 1;
    }
    
    if (checkCoin && checkBill){
        //coin detection
        CoinDetection coinDetection;
        MoneyDetection coinsDetected = coinDetection.detectCoins(imageOriginal, Mat());
        
        //bill detection
        BillDetection billDetection;
        MoneyDetection billsDetected = billDetection.detectBill(imageOriginal, coinsDetected.identifiedMoneyImage);//draw the identified bills on top of the drawn identified coins
            
        //write total money text in the image with identified elements
        int total = billsDetected.totalValue + coinsDetected.totalValue;
        Mat drawnImage = getSquareImage(billsDetected.identifiedMoneyImage, 600);
        putText(
            drawnImage, 
            "Coins: " + to_string(coinsDetected.nElements) + "; Bills: " + to_string(billsDetected.nElements) + " Total Money:" + to_string(total) + " euros", 
            Point(drawnImage.cols / 10, drawnImage.rows - drawnImage.rows / 10), 
            FONT_HERSHEY_COMPLEX_SMALL, 
            0.7, 
            Scalar(0, 255, 255), 
            0.6, 
            CV_AA
        );

        //show image with identified money and value
        string wname = "Money Recognition - result";
        namedWindow(wname, WINDOW_NORMAL);
        resizeWindow(wname, 800, 800);
        imshow(wname, drawnImage);
        waitKey();

        return 0;
    }
    else if (checkCoin && !checkBill){
        //check coin (only)
        CoinDetection coinDetection;
        MoneyDetection coinsDetected = coinDetection.detectCoins(imageOriginal, Mat());
        Mat drawnImage = coinsDetected.identifiedMoneyImage;
        putText(
            drawnImage, 
            "Coins: " + to_string(coinsDetected.nElements) + "; Total Money:" + to_string(coinsDetected.totalValue) + " euros", 
            Point(drawnImage.cols / 10, drawnImage.rows - drawnImage.rows / 10), 
            FONT_HERSHEY_COMPLEX_SMALL, 
            0.7, 
            Scalar(0, 255, 255), 
            0.6, 
            CV_AA
        );

        //show image with identified money and value
        string wname = "Coin Recognition - result";
        namedWindow(wname, WINDOW_NORMAL);
        resizeWindow(wname, 800, 800);
        imshow(wname, drawnImage);
        waitKey();

        return 0;
    }
    else if (checkBill && !checkCoin){
        //check bill (only)
        BillDetection billDetection;
        MoneyDetection billsDetected = billDetection.detectBill(imageOriginal, Mat());
        Mat drawnImage = getSquareImage(billsDetected.identifiedMoneyImage, 600);
        putText(
            drawnImage, 
            "Bills: " + to_string(billsDetected.nElements) + "; Total Money:" + to_string(billsDetected.totalValue) + " euros", 
            Point(drawnImage.cols / 10, drawnImage.rows - drawnImage.rows / 10), 
            FONT_HERSHEY_COMPLEX_SMALL, 
            0.7, 
            Scalar(0, 255, 255), 
            0.6, 
            CV_AA
        );

        //show image with identified money and value
        string wname = "Bill Recognition - result";
        namedWindow(wname, WINDOW_NORMAL);
        resizeWindow(wname, 800, 800);
        imshow(wname, drawnImage);
        waitKey();

        return 0;
    }
    return 1;
}