# Money Recognition
An OpenCV program that recognizes euro bills and coins from images. Opencv 3 was used for the development of this project.

## To compile only the coin detector program:
```
g++ CoinDetector.cpp -o <appname>.out `pkg-config --cflags --libs opencv`
```

## To compile only the bill detector program:
```
g++ billDetector.cpp.cpp -o <appname>.out `pkg-config --cflags --libs opencv`
```

## To use both programs, compile the 'main.cpp':
```
g++ main.cpp -o <appname>.out `pkg-config --cflags --libs opencv`
```

## To run
```
Usage: ./app <image file> [-c | -b]
-c will detect only coins. -b will detect only bills.
To detect both coins and bills insert only the image as an argument.
```

### The bill detector program considers recent euro bills ('Europa' series).
### The coin detector program considers that a 2 euro coin exist in the picture to scan in order to obtain a measurement scale