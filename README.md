# Money Recognition
An OpenCV program written in C++ that recognizes euro bills and coins from images. Opencv 3 was used for the development of this project.

## To run
```
Usage: ./app <image file> [-c | -b]
-c will detect only coins. -b will detect only bills.
To detect both coins and bills insert only the image as an argument.
```

### The coin detector module considers that a 2 euro coin exist in the picture to scan in order to obtain a measurement scale
### The bill detector module considers that the note(s) are on a non light, smooth surface and that the note are horizontal, non overlapping each other.
