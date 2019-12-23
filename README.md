# Money Recognition
An OpenCV program that recognizes euro bills and coins from images

To build:
g++ CoinDetector.cpp -o CoinDetector.out `pkg-config --cflags --libs opencv`

To run:
./CoinDetector.out <image>