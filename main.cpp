#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect.hpp>
#include <string>

#include <iostream>

using namespace cv;
using namespace std;

struct MoneyDetection {
    cv::Mat identifiedMoneyImage;
    float totalValue;
    int nElements;
};

/////////////////////     Helper functions  ///////////////////////////

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

//helper function to detect and remove rectangles identified in the oberder of the image
bool noRectangleOverImage(Mat img, vector<Point> rect){
    double imgHeight = img.size().height;
    double imgWidth = img.size().width;
    if(find(rect.begin(), rect.end(), Point(0, 0)) != rect.end()) {
        //contains
        return false;
    }
    return true;
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
        vector<Mat> notesImages = cropBills(image, imageToDraw, rects);
        for( size_t i = 0; i < notesImages.size(); i++ ){
            Mat noteImage = notesImages[i];
            //noteImage = quantizeImage(noteImage, 4);
            string name = "note ";
            name+= to_string(i);
            //cout <<name<<endl;
            Scalar hsvColor = getDominantHSVColor(noteImage);
            int value = classifyBill(hsvColor);
            cout <<"detected: " << to_string(value)<< " euros" <<endl;
            drawOnImage(imageToDraw, rects[i], to_string(value));
            
            total += value;

            namedWindow(name, WINDOW_NORMAL);
            resizeWindow(name, 800, 800);
            imshow(name, noteImage);
        }
        MoneyDetection moneyDetect = {
            .identifiedMoneyImage = imageToDraw,
            .totalValue = total,
            .nElements = notesImages.size()
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

    int classifyBill(Scalar hsv){
        Scalar HSV500Lower = Scalar(135, 60, 20); //pink
        Scalar HSV500Upper = Scalar(165, 255, 255); //pink color

        Scalar HSV200Lower = Scalar(10, 70, 20); //yellow color
        Scalar HSV200Upper = Scalar(32, 200, 255); //yellow color

        Scalar HSV100Lower = Scalar(40, 30, 20); //green color
        Scalar HSV100Upper = Scalar(65, 255, 255); //green color

        Scalar HSV50Lower = Scalar(5, 60, 20); //orange color
        Scalar HSV50Upper = Scalar(22, 220, 255); //orange color

        Scalar HSV20Lower = Scalar(85, 0, 20); //blue - gray  color
        Scalar HSV20Upper = Scalar(120, 220, 255); //blue color
        
        Scalar HSV10Lower = Scalar(0, 50, 20); //red  color
        Scalar HSV10Upper = Scalar(10, 220, 255); //red color
        //red color is between 175 - 10 in H scale
        Scalar HSV10Lower_2 = Scalar(175, 50, 20); //red  color
        Scalar HSV10Upper_2 = Scalar(180, 220, 255); //red color

        Scalar HSV5Lower = Scalar(8, 0, 20);  //grayish green
        Scalar HSV5Upper = Scalar(100, 50, 255);  //grayish green

        if (numberInRange(HSV500Lower[0], HSV500Upper[0], hsv[0]) && 
                numberInRange( HSV500Lower[1], HSV500Upper[1], hsv[1]) &&
                numberInRange( HSV500Lower[2], HSV500Upper[2], hsv[2])){
            return 500;
        }
        else if (numberInRange(HSV200Lower[0], HSV200Upper[0], hsv[0]) && 
                numberInRange( HSV200Lower[1], HSV200Upper[1], hsv[1]) &&
                numberInRange( HSV200Lower[2], HSV200Upper[2], hsv[2])){
            return 200;
        }
        else if (numberInRange(HSV100Lower[0], HSV100Upper[0], hsv[0]) && 
                numberInRange( HSV100Lower[1], HSV100Upper[1], hsv[1]) &&
                numberInRange( HSV100Lower[2], HSV100Upper[2], hsv[2])){
            return 100;
        }
        else if (numberInRange(HSV50Lower[0], HSV50Upper[0], hsv[0]) && 
                numberInRange( HSV50Lower[1], HSV50Upper[1], hsv[1]) &&
                numberInRange( HSV50Lower[2], HSV50Upper[2], hsv[2])){
            return 50;
        }
        else if (numberInRange(HSV20Lower[0], HSV20Upper[0], hsv[0]) && 
                numberInRange( HSV20Lower[1], HSV20Upper[1], hsv[1]) &&
                numberInRange( HSV20Lower[2], HSV20Upper[2], hsv[2])){
            return 20;
        }
        else if ( ( numberInRange(HSV10Lower[0], HSV10Upper[0], hsv[0]) && 
                numberInRange( HSV10Lower[1], HSV10Upper[1], hsv[1]) &&
                numberInRange( HSV10Lower[2], HSV10Upper[2], hsv[2]) ) || numberInRange(HSV10Lower_2[0], HSV10Upper_2[0], hsv[0]) && 
                numberInRange( HSV10Lower_2[1], HSV10Upper_2[1], hsv[1]) &&
                numberInRange( HSV10Lower_2[2], HSV10Upper_2[2], hsv[2])){
            return 10;
        }
        else if (numberInRange(HSV5Lower[0], HSV5Upper[0], hsv[0]) && 
                numberInRange( HSV5Lower[1], HSV5Upper[1], hsv[1]) &&
                numberInRange( HSV5Lower[2], HSV5Upper[2], hsv[2])){
            return 5;
        }
        return 0;
    }

    int numberInRange(int lower, int upper, int n){
        return ( n <= upper && n >= lower );
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
        int thresh = 50, N = 11;

        
        Mat blurred;
        int k = 1;
        for (int c = 0; c < 6; c++){
            k+=2;
            //blurr to remove noise. Several ksizes are experimented to get the best detection of rectangles
            medianBlur(image, blurred, k);

            Mat gray0(blurred.size(), CV_8U), gray, timg, pyr;

            vector<vector<Point> > contours;

            //convert to gray scale
            cvtColor(blurred, gray0, CV_BGR2GRAY);

            Mat gray1;
            gray0.copyTo(gray1);

            //first: use canny then extract rectangle contours, then use adaptive threshold and extract rectangle contours.
            for( int l = 0; l < 2; l++ )
            {
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
                else{
                    adaptiveThreshold(gray0, gray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,11,2);
                    //wee need black as blackground
                    int nWhite = countNonZero(gray);
                    int imageTotalPixels = gray.size().width * gray.size().height;
                    if (nWhite > imageTotalPixels/2){
                        //bigger number of white: white as background, apply inverse binary theshold
                        adaptiveThreshold(gray0, gray, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,11,2);
                    }
                    //int morph_size = 3;
                    //Mat element = getStructuringElement( MORPH_RECT, Size( 4*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
                    dilate(gray, gray, Mat(), Point(-1,-1),1);
                }

                //for debug                
                string name = "median blurred ";
                name+= ", ";
                name+= to_string(l);
                namedWindow(name, WINDOW_NORMAL);
                resizeWindow(name, 800, 800);
                imshow(name, gray);
                
                // find contours and store them all as a list
                //(findContours treads white as foreground and black as background:) (https://answers.opencv.org/question/2885/findcontours-gives-me-the-border-of-the-image/)
                findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); //CV_RETR_EXTERNAL -> dont detect any inner squares inside the bills!

                vector<Point> approx;

                // test each contour
                for( size_t i = 0; i < contours.size(); i++ )
                {
                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);

                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex. 10 000 is a area we find acceptable as it filters little rectangles, such as the ones inside bills with the euro flag
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation
                    if( approx.size() == 4 && fabs(contourArea(approx)) > 10000 &&
                        isContourConvex(approx) ) {
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
                        if( maxCosine < 1.2 && noRectangleOverImage(gray, approx)){
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
        groupRectangles(rects, 1, 0.08); //we may have multiple rectangles really close, so we merge them together. at least 2 triangles must be merged and they must be really overlaped to be merged
        return rects;
    }

    static Scalar getDominantHSVColor(Mat image){
        Mat1b mask(image.size(), uchar(0));
        Scalar meanIntensity = mean(image);
        //cout << meanIntensity << endl;
        Scalar hsv = ScalarBGR2HSV(meanIntensity);
        //cout << hsv << endl;
        return hsv;
    }

    static Mat cropHalfImage(Mat image){
        //crop half left
        //Mat croppedFrame = image(Rect(0, 0, image.cols/2, image.rows));
        //crop half right
        Mat croppedFrame = image(Rect(image.cols/2, 0, image.cols/2, image.rows));
        return croppedFrame;
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
    
    // the function draws all the squares in the image using rectangles over the identified bills and adds text over the note with the identified value
    void drawOnImage( Mat& imageToDraw, Rect rect, string text ){
        rectangle(imageToDraw, rect, Scalar(0,255,0), 3, LINE_AA); //draw rectangle surrounding the note
        if (text == "0"){
            text = "?";
        } 
        putText(imageToDraw, text, Point(rect.x, rect.y), FONT_HERSHEY_COMPLEX_SMALL, 3.0, 
            Scalar(0, 0, 255), 
            3, 
            CV_AA);
    }

    //given an image and a vector of Rects whose coordinates correspond to the location of a bill in the image, crop that bill as an image
    vector<Mat> cropBills( Mat image, Mat& imageToDraw, const vector<Rect>& squares ){
        vector<Mat> notesImages;
        for( size_t i = 0; i < squares.size(); i++ ){
            Rect rect = squares[i];
            Mat noteImage(image, rect);
            //crop a little the image to exclude any possible background of the bill
            noteImage = cropImage(noteImage, 20);
            noteImage = cropHalfImage(noteImage);
            notesImages.push_back(noteImage);
        }
        return notesImages;
    }
};


class CoinDetection 
{ 

    /*
    There is 3 categories:
        - Pennies (color is hgsv(<= 13, 130-190, 60-110) then 5/2/1 cents)
        - Big cents (color is hgsv(15-20, 54-127, 85-207) then 10/20/50 cents)
        - Euros (color is hgsv(18-19, 110-160, 95-190) then 1/2 euros)	
    */
    private:
    
    bool isEuro(Mat img, Vec3f circ) {
        Scalar hsv = getMeanCircleHSV(img, circ);
        return hsv[0] >= 15 && hsv[0] < 18 && hsv[1] > 50 && hsv[1] <=130 && hsv[2] > 85 && hsv[2] <=210;
    }

    bool isPenny(Mat img, Vec3f circ) {
        circ[2] = 5;
        Scalar hsv = getMeanCircleHSV(img, circ);
        return hsv[0] >=8 && hsv[0] <=15 && hsv[1] >= 140 && hsv[1] <=200 && hsv[2] >= 60 && hsv[2] <=135;
    }

    bool isMidCent(Mat img, Vec3f circ) {
        circ[2] = 5;
        Scalar hsv = getMeanCircleHSV(img, circ);
        return hsv[0] >= 16 && hsv[0] <=20 && hsv[1] >= 110 && hsv[1] <=175 && hsv[2] >= 90 && hsv[2] <=205;
    }

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

    public: 
    // Given an image, returns in a struct the original image with drawings surrounding the identified coins, total ammount and number of coins.
    //second argument is an image to draw circles surrounding the identified elements. If empty, the original image is used to draw
    MoneyDetection detectCoins(Mat img, Mat imageToDraw){
        Mat gray, imgScaled;
        
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

        Canny(gray, edgesOpened, 50, 170, 3);
        
        
        //threshold(gray, edges, 100, 255, CV_THRESH_OTSU);
        //adaptiveThreshold(edges, edges, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 1);
        //Morphology
        //Dilation
        int dilation_type = MORPH_RECT; // dilation_type = MORPH_RECT,MORPH_CROSS,MORPH_ELLIPSE
        int dilation_size = 3;  
        Mat element = getStructuringElement(
            dilation_type,
            Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            Point(dilation_size, dilation_size)
        );
        morphologyEx(edgesOpened, edges, MORPH_CLOSE, element);
        

        namedWindow("Canny", 0);
        resizeWindow("Canny", 600, 600);
        imshow("Canny", edges);


        
        vector<Vec3f> allCircles, circles;
        
        //HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 2, gray.rows / 12, 200, 80, 20, 70);

        // Detects cricles from canny image. param 1 and param 2 values picked based on trial and error.
        HoughCircles(edges, allCircles, CV_HOUGH_GRADIENT, 2, gray.rows / 12, 200, 80, 20, 70);

        // Filter circles
        copy_if (
            allCircles.begin(), 
            allCircles.end(), 
            back_inserter(circles), 
            [&imgScaled, this](Vec3f circle){
                return  isPenny(imgScaled, circle) == true ||
                        isEuro(imgScaled, circle) == true ||
                        isMidCent(imgScaled, circle) == true;
            } 
        );

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

        if (circles.size() == 0){
            MoneyDetection moneyDetect = {
            .identifiedMoneyImage = imageToDraw,
            .totalValue = 0,
            .nElements = 0
            };
            return moneyDetect;
        }
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
            
            Vec3f newCirc = circles[i];
            newCirc[2] = 5;
            Scalar hsv = getMeanCircleHSV(imgScaled, newCirc);

            /*
            Average colors were picked based on trial and error. 		
            In each color category we differentiate coins based on the area of each coin compared to the 2 euro coin.
            Areas were picked based on trial and error
            */

            if (isPenny(imgScaled, circles[i]))
            {
                if (ratio >= 0.70)
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
                else if ((ratio >= 0.55) && (ratio<.70))
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
                else if ((ratio >= 0.3) && (ratio<.55))
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
            else if (isEuro(imgScaled, circles[i]))
            {
                if (ratio >= 0.90)
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
                else if ((ratio >= 0.40) && (ratio<90))
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
            else if (isMidCent(imgScaled, circles[i]))
            {
                if (ratio >= 0.80)
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
                else if ((ratio >= 0.60) && (ratio<.80))
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
                else if ((ratio >= 0.40) && (ratio<.60))
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
            /*
            drawResult(
                imageToDraw, 
                center, 
                radius, 
                to_string(i)+ "-?? euro"
            );
            cout <<  i << " ?? cents - " << hsv <<endl;
            */

        }

        MoneyDetection moneyDetect = {
            .identifiedMoneyImage = imageToDraw,
            .totalValue = change,
            .nElements = coins
        };
        return moneyDetect;
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
        
        //bill detection
        BillDetection billDetection;
        MoneyDetection billsDetected = billDetection.detectBill(imageOriginal, Mat());//draw the identified bills on top of the drawn identified coin
        
        //coin detection
        CoinDetection coinDetection;
        MoneyDetection coinsDetected = coinDetection.detectCoins(imageOriginal, getSquareImage(billsDetected.identifiedMoneyImage, 600));

        //write total money text in the image with identified elements
        int total = billsDetected.totalValue + coinsDetected.totalValue;
        Mat drawnImage = coinsDetected.identifiedMoneyImage;
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