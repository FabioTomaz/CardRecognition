#ifndef MONEY_DETECT_H
#define MONEY_DETECT_H

#include <iostream>
#include <opencv2/opencv.hpp>

struct MoneyDetection {
    cv::Mat identifiedMoneyImage;
    int totalValue;
};

MoneyDetection detectBill(cv::Mat image);

MoneyDetection detectCoins(cv::Mat image);

#endif