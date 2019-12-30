#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace cv;
using namespace std;

Scalar ScalarHSV2BGR(Scalar scalar);

Mat getSquareImage( const cv::Mat& img, int target_width);

void drawResult(Mat img, Point center, float radius, string text);

int main(int argc, char** argv)
{
	Mat img, imgScaled, gray;
	img = imread(argv[1]);
	if (img.empty()) {
		cout << "Wrong file or unexistant!" <<endl;
		return -1; 
	}

	imgScaled = getSquareImage(img, 600);

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

	// Double check for the circles - Just the edge image at this moment produces a lot of false circles - when the Hough circles function is run
	// Shortlisting good circle candidates by doing a contour analysis
	vector<vector<Point>> contours, contoursfil;
	vector<Vec4i> hierarchy;
/*	Mat contourImg2 = Mat::ones(edges.rows, edges.cols, edges.type());

	//Find all contours in the edges image
	findContours(edges.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	for (int j = 0; j < contours.size(); j++)
	{
		//Only give me contours that are closed (i.e. they have 0 or more children : hierarchy[j][2] >= 0) AND contours that dont have any parent (i.e. hierarchy[j][3] < 0 )
		if ((hierarchy[j][2] >= 0) && (hierarchy[j][3] < 0))
		{
			contoursfil.push_back(contours[j]);
		}
	}

	for (int j = 0; j < contoursfil.size(); j++)
	{
		drawContours(contourImg2, contoursfil, j, CV_RGB(255, 255, 255), 1, 8);
	}
	namedWindow("Contour Image Filtered", WINDOW_NORMAL);
	resizeWindow("Contour Image Filtered", 600, 600);
	imshow("Contour Image Filtered", contourImg2);*/

	// good values for param-2 - for a image having circle contours = (75-90) ... for the edge image - (100-120)
	vector<Vec3f> circles;

	HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 2, gray.rows / 9, 200, 100, 25, 90);

	//struct to sort the vector of pairs <int,double> based on the second double value
	struct sort_pred {
		bool operator()(const Vec3f &left, const Vec3f &right) {
			return left[2]< right[2];
		}
	};
	//sort in descending
	std::sort(circles.rbegin(), circles.rend(), sort_pred());

	float largestRadius = circles[0][2];
	float change = 0;
	int coins = 0;
	float ratio;

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		float radius = circles[i][2];
		ratio = ((radius*radius) / (largestRadius*largestRadius));
	    
		Mat1b mask(imgScaled.size(), uchar(0));
    	circle(mask, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(255), CV_FILLED);
		Scalar meanIntensity = mean(imgScaled, mask);
		Scalar hsv = ScalarHSV2BGR(meanIntensity);
/*
		drawResult(
			imgScaled, 
			center, 
			radius, 
			to_string(i)+ "-?? euro"
		);

		cout <<  i << " ?? cents - " << hsv <<endl;*/

		// If <= 13, 130-190, 60-100  then 5/2/1 cents
		// Using an area ratio based discrimination
		if (hsv[0] <=13 && hsv[1] >= 130 && hsv[1] <=190 && hsv[2] >= 60 && hsv[2] <=110){
			if ((ratio >= 0.75) && (ratio<.95))
			{
				drawResult(
					imgScaled, 
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
					imgScaled, 
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
					imgScaled, 
					center, 
					radius, 
					to_string(i)+ "-1 cent"
				);
				change = change + .01;
				coins = coins + 1;
				cout <<  i << " 1 cent - " << hsv <<endl;
			}
			continue;			
		} 

		if (hsv[0] >= 15 && hsv[0] < 18 && hsv[1] > 50 && hsv[1] <=130 && hsv[2] > 85 && hsv[2] <=210){
			//15-20 54-127 85-207
			if (ratio >= 0.85)
			{
				drawResult(
					imgScaled, 
					center, 
					radius, 
					to_string(i)+ "-2 euro"
				);
				change = change + 2.0;
				coins = coins + 1;
				cout << i << "2 euro - " << hsv <<endl;
			}
			else if ((ratio >= 0.40) && (ratio<80))
			{
				drawResult(
					imgScaled, 
					center, 
					radius, 
					to_string(i)+ "-1 euro"
				);
				change = change + 1.00;
				coins = coins + 1;
				cout << i << " 1 euro - " << hsv <<endl;
			}
			continue;
		}
		
		if (hsv[0] >= 18 && hsv[0] <=20 && hsv[1] >= 110 && hsv[1] <=160 && hsv[2] >= 95 && hsv[2] <=190){
			// 18-19, 110-160, 95-190
			if (ratio >= 0.90)
			{
				drawResult(
					imgScaled, 
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
					imgScaled, 
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
					imgScaled, 
					center, 
					radius, 
					to_string(i)+ "-10 cents"
				);
				change = change + .10;
				coins = coins + 1;
				cout << i << " 10 cents - " << hsv <<endl;
			}
		} 
		continue;
	}

	putText(
		imgScaled, 
		"Coins: " + to_string(coins) + " // Total Money:" + to_string(change), 
		Point(imgScaled.cols / 10, imgScaled.rows - imgScaled.rows / 10), 
		FONT_HERSHEY_COMPLEX_SMALL, 
		0.7, 
		Scalar(0, 255, 255), 
		0.6, 
		CV_AA
	);

	namedWindow("Result", WINDOW_NORMAL);
	resizeWindow("Result", 600, 600);
	imshow("Result", imgScaled);

	waitKey();

	return 0;
}

Scalar ScalarHSV2BGR(Scalar scalar) {
    Mat hsv;
    Mat rgb(1,1, CV_8UC3, scalar);
    cvtColor(rgb, hsv, CV_BGR2HSV);
    return Scalar(hsv.data[0], hsv.data[1], hsv.data[2]);
}

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