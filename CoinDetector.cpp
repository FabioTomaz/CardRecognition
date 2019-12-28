#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace cv;
using namespace std;

float getCircularityThresh(vector<Point> cntr);



Mat getSquareImage( const cv::Mat& img, int target_width = 600 )
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

int main(int argc, char** argv)
{
	Mat img, imgScaled, gray;
	img = imread(argv[1]);
	if (img.empty()) {
		cout << "Wrong file or unexistant!" <<endl;
		return -1; 
	}

	imgScaled = getSquareImage(img);

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

	float circThresh;

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

	HoughCircles(edges, circles, CV_HOUGH_GRADIENT, 2, gray.rows / 9, 200, 100, 25, 100);

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
	float ratio;

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		float radius = circles[i][2];
		// draw the circle center
		circle(imgScaled, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		circle(imgScaled, center, radius, Scalar(0, 0, 255), 2, 8, 0);
		rectangle(
			imgScaled, 
			Point(center.x - radius - 5, center.y - radius - 5), 
			Point(center.x + radius + 5, center.y + radius + 5), 
			CV_RGB(0, 0, 255), 
			1, 
			8, 
			0
		); //Opened contour - draw a green rectangle arpund circle
		ratio = ((radius*radius) / (largestRadius*largestRadius));

		//Using an area ratio based discrimination .. after some trial and error with the diff sizes ... this gives good results.
		if (ratio >= 0.95)
		{
			putText(imgScaled, "2 Euro", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + 2.0;
		}
		else if ((ratio >= 0.70) && (ratio<95))
		{
			/*putText(imgScaled, "50 cents", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + 1.0;*/
			putText(imgScaled, "1 euro", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .50;
		}
		else if ((ratio >= 0.50) && (ratio<.70))
		{
			putText(imgScaled, "20-cents", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .20;
		}
		else if ((ratio >= 0.40) && (ratio<.50))
		{
			putText(imgScaled, "10-cents", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .10;
		}
		else if ((ratio >= 0.20) && (ratio<.40))
		{
			putText(imgScaled, "5-cents", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .05;
		}
		else if ((ratio >= 0.10) && (ratio<.20))
		{
			putText(imgScaled, "2-cents", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .02;
		}
		else if ((ratio >= 0.0) && (ratio<.10))
		{
			putText(imgScaled, "1-cent", Point(center.x - radius, center.y + radius + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.4, CV_AA);
			change = change + .01;
		}

	}

	putText(imgScaled, "Total Money:" + to_string(change), Point(imgScaled.cols / 10, imgScaled.rows - imgScaled.rows / 10), FONT_HERSHEY_COMPLEX_SMALL, 0.7, Scalar(0, 255, 255), 0.6, CV_AA);

	namedWindow("circles", WINDOW_NORMAL);
	resizeWindow("circles", 600, 600);
	imshow("circles", imgScaled);

	waitKey();

	return 0;
}

float getCircularityThresh(vector<Point> cntr)
{
	float perm, area;

	perm = arcLength(Mat(cntr), true);
	area = contourArea(Mat(cntr));

	return ((perm*perm) / area);

}

/*
approx = cv.approxPolyDP(cnt, 0.008 * cv.arcLength(cnt, True), True)
            area = cv.contourArea(cnt)
            if len(approx) > 12 and area > 750:
                (x, y), radius = cv.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                if area > 8000:
                    real_contoursB.append((center, radius))
                    #cv.circle(original, center, radius, (255, 255, 0), 4)
                else:
                    real_contoursG.append((center, radius))
                    #cv.circle(original, center, radius, (0, 255, 0), 4)
*/

