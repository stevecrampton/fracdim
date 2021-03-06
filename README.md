# fracdim

The fractal dimension of an object is a measure of how similar the object is to itself at different scales. For instance, many objects in nature have patterns that occur when you look at an object from afar; these patterns sometimes reoccur when you zoom in and look more closely at an object. Objects with these characteristics are called fractals.

fracdim.py estimates the fractal dimension of an object. For purposes of the program, the object must be represented as a set of points. The set of points can be contained in a csv file (a common spreadsheet and data format) or in an image. If an image file is provided, the program attempts to find the points in the image that represent the fractal automatically. Because such an approach is necessarily difficult, it is best to supply an image that has been thresholded in advance to show where the fractal pattern is. Most image-processing programs, such as the Gnu Image-Manipulation Program, have a thresholding feature.

A fractal dimension of 1.0 indicates that there are apparently no patterns that repeat at different scales. A fractal dimension greater than 1.0 (but less than 2.0) indicates some degree of self-similarity. A dimension of 2.0 indicates that the object is a 2-dimensional object, for example, a plane.

The program finds what is called the "box-counting dimension" using a Monte Carlo algorithm. As each level is processed, the boxes are displayed.

Also included are data files to use with the program.
