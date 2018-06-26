
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;

#define N 3	      // kernel size 
#define THRESHOLD 30  
#define TH_RING 70
#define PI 3.14

//#define FACE 1

/** Function Headers */
vector<Rect> detectAndDisplay( Mat frame );

float F1_score(vector<Rect> detected, vector<Rect> label);

// to compute x and y derivative --- df/dx and df/dy
void derivative(cv::Mat &image, cv::Mat &output, double kernel[N][N]);

// to compute the magnitude of the gradient
void gradientMagnitude(cv::Mat &image, cv::Mat &output, cv::Mat &xDer, cv::Mat &yDer);

// to compute the direction of the gradient
void gradientDirection(cv::Mat &input, cv::Mat &output, cv::Mat &xDer, cv::Mat &yDer);

// to threshold the image
void threshold(cv::Mat &input, cv::Mat &output, int threshold);

// Hough transform for circle detection
void houghTransform(cv::Mat &th_magnitude, cv::Mat &orientation, int threshold, cv::Mat &hough_image);

// Viola-Jones combined with Circle detection
void violaAndCircle(Mat &original, Mat& viola_circle);

// Viola-Jones combined with Line detection
void violaAndLine(Mat &original, Mat& viola_line);

// Hough transform to the whole image, to produce the 2-D representation of the Hough space and the thresholded magnitude
void hough_to_whole_image(Mat &frame, string imageName);

// detect lines and display them on the image
vector<Vec4i> detectAndDisplayLine(Mat& image, Mat& th_magnitude, Mat& hough_line);

// compute the number of line intersections
int intersections(vector<Vec4i> lines);

// Viola-Jones combined with Circle detecion and Line detection
vector<Rect> violaLineRing(Mat& frame, Mat& viola_line_ring);

// Hough transform for line detection
Mat lineDetection(Mat &original, Mat &magnitude);

// final dartboard detector
vector<Rect> improvedDetector(Mat& frame, Mat& result);

// compute the difference between black and white values in the image (histogram)
int difference(Mat& image);

// initialize the ground truth from the file system
vector<Rect> initialize(string filename);

// check if 'value' is between 'min' and 'max'
bool valueInRange(int value, int min, int max);

// check if two boxes overlap
bool rectOverlap(Rect A, Rect B);

// compute the area of A intersecate B, where A and B are two boxes
int areaOfIntersection(Rect A, Rect B);

// split boxes in different groups based on the overlapping
vector< vector<Rect> > splitInGroups(vector<Rect> detected);

// compute the average box of many overlapping boxes
vector<Rect> averageBoxes(vector< vector<Rect> > groups);

/** Global variables */
String cascade_name           = "frontalface.xml";           // face classifier
String cascade_name_dartboard = "./dartcascade/cascade.xml"; // dartboard classifier
CascadeClassifier cascade;

 // kernel for x-derivative
double dfdx[N][N] = {-1.0, 0.0, 1.0,		\
		     -2.0, 0.0, 2.0,		\
		     -1.0, 0.0, 1.0};

// kernel for y-defivative
double dfdy[N][N] = {-1.0, -2.0, -1.0,					\
		     0.0, 0.0, 0.0,					\
		     1.0, 2.0, 1.0};




/** @function main */
int main( int argc, const char** argv ){
  string imageName(argv[1]);
  string fileName = "dartlabel/" + imageName + ".txt";;
  string cascadeName = cascade_name_dartboard;
  
#ifdef FACE
  fileName = "facelabel/" + imageName + ".txt";
  cascadeName = cascade_name;
#endif

  // 1. Read Input Image
  Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

  // 2. Load the Strong Classifier in a structure called `Cascade'
  if( !cascade.load( cascadeName ) ){
    printf("--(!)Error loading\n");
    return -1; 
  }  

  // 3. initialize labels
  vector<Rect> label = initialize(fileName);

  // 4. detect objects
  Mat detected;
  vector<Rect> finalBoxes = improvedDetector(frame, detected);
  
  // 5. compute F1-score
  float score = F1_score(finalBoxes, label);
  printf("F1-score = %f\n", score);
  
  // 6. output image
  imwrite("detected.jpg", detected);
  
  return 0;
}

/** @function detectAndDisplay */
std::vector<Rect> detectAndDisplay( Mat frame ){
  std::vector<Rect> obj;
  Mat frame_gray;
  
  // 1. Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );
  
  // 2. Perform Viola-Jones Object Detection 
  cascade.detectMultiScale( frame_gray, obj, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
  
  // 3. Draw box around dartboard found
  for( unsigned int i = 0; i < obj.size(); i++ ){
    rectangle(frame, Point(obj[i].x, obj[i].y), Point(obj[i].x + obj[i].width, obj[i].y + obj[i].height), Scalar(0,255,0), 2);
  }
  return obj;
}

void houghTransform(cv::Mat &th_magnitude, cv::Mat &orientation, int threshold, cv::Mat &hough_image){
  int max_radius = min((orientation.rows)/2, (orientation.cols)/2);
  int n = th_magnitude.size[0];
  int m = th_magnitude.size[1];
  
  // create and initialise to 0 the Hough space
  int*** Hspace = (int***) malloc(sizeof(int*) * n);
  for(int i = 0; i < n; i++){
    Hspace[i] = (int**) malloc(sizeof(int*) * m);
    for(int j = 0; j < m; j++){
      Hspace[i][j] = (int*) calloc(max_radius, sizeof(int));
    }
  } 
  
  int x0,y0;
  // compute the hough space
  for ( int i = 0; i < th_magnitude.rows; i++ ){	
    for( int j = 0; j < th_magnitude.cols; j++ ){
      // if the value is greater than the threshold
      if(th_magnitude.at<uchar>(i,j) > threshold){
	// for each radius
	for (int r = 0; r < max_radius; r++){
	  double p = orientation.at<uchar>(i,j) * 2 * PI / 255;
	  x0 = i + r * sin(p);
	  y0 = j + r * cos(p);
	  if(x0 < n && x0 >= 0 && y0 < m && y0 >= 0){ // check the boundaries
	    Hspace[x0][y0][r]++;
	  }
	  
	  x0 = i - r * sin(p);
	  y0 = j - r * cos(p);
	  if(x0 < n && x0 >= 0 && y0 < m && y0 >= 0){
	    Hspace[x0][y0][r]++;
	  }
	}
	
      }
    }
  }
  
  // compute the max value over all the sum of the radius, to scale between 0 and 255
  float maxValue = 0;
  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++){
      int sum = 0;
      for(int k = 0; k < max_radius; k++)
	sum += Hspace[i][j][k];
      maxValue = sum > maxValue ? sum : maxValue;
    }

  // convert hough space into a 2D image
  hough_image.create(th_magnitude.size(), th_magnitude.type());
  for (int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      float sumRadius = 0;
      for(int r = 0; r < max_radius; r++){
	sumRadius += Hspace[i][j][r]; 
      }
      // normalize the result 
      uchar val = (uchar) ((sumRadius / maxValue) * 255);
      hough_image.at<uchar>(i,j) = val; 
    }
  }
  
}

void gradientDirection(cv::Mat &input, cv::Mat &output, cv::Mat &xDer, cv::Mat &yDer){
  // intialise the output using the input
  output.create(input.size(), input.type());
  
  for ( int i = 0; i < input.rows; i++ ){	
    for( int j = 0; j < input.cols; j++ ){
      double xVal = xDer.at<uchar>(i,j) - 128;
      double yVal = yDer.at<uchar>(i,j) - 128;
      double val = atan(yVal/ xVal);
      output.at<uchar>(i, j) = val * 255 / (2*PI);
    }
  }
}

void gradientMagnitude(cv::Mat &input, cv::Mat &output, cv::Mat &xDer, cv::Mat &yDer){
  // intialise the output using the input
  output.create(input.size(), input.type());
  
  for ( int i = 0; i < input.rows; i++ ){	
    for( int j = 0; j < input.cols; j++ ){
      double xVal = xDer.at<uchar>(i,j) - 128;
      double yVal = yDer.at<uchar>(i,j) - 128;
      double val = sqrt((xVal * xVal) + (yVal * yVal));
      output.at<uchar>(i, j) = val;
    }
  }
}

void derivative(cv::Mat &input, cv::Mat &output, double kernel[N][N]){
  // intialise the output using the input
  output.create(input.size(), input.type());
  
  int kernelRadiusX = ( N - 1 ) / 2;
  int kernelRadiusY = ( N - 1 ) / 2;
  
  cv::Mat paddedInput;
  // Forms a border around an image.
  cv::copyMakeBorder( input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv::BORDER_REPLICATE );
  
  // now we can do the convoltion
  for ( int i = 0; i < input.rows; i++ ){	
    for( int j = 0; j < input.cols; j++ ){
      double sum = 0.0;
      for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ){
	for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ){
	  // find the correct indices we are using
	  int imagex = i + m + kernelRadiusX;
	  int imagey = j + n + kernelRadiusY;
	  int kernelx = m + kernelRadiusX;  
	  int kernely = n + kernelRadiusY; 
	  
	  // get the values from the padded image and the kernel
	  int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
	  double kernalval = kernel[kernelx][kernely]; 
	  
	  // do the multiplication
	  sum += imageval * kernalval; 
	  
	}
      }
      // set the output value as the sum of the convolution
      output.at<uchar>(i, j) = (sum / 8) + 128;
    }
  }  
  
}

void threshold(cv::Mat &input, cv::Mat &output, int threshold){
  // intialise the output using the input
  output.create(input.size(), input.type());
  
  // threshold the image
  for ( int i = 0; i < input.rows; i++ ) 	
    for( int j = 0; j < input.cols; j++ )
      output.at<uchar>(i, j) = input.at<uchar>(i,j) >= threshold ? 255 : 0;
}


float F1_score(vector<Rect> detected, vector<Rect> label){
  if( label.size() == 0)
    return 0;
  
  int* checked = (int*) calloc(label.size(), sizeof(int)); // detected times per each labelled box
  float percentage = 0.4;
  
  for(unsigned int i = 0; i < detected.size(); i++){
    Rect A = detected[i];
    for( unsigned int j = 0; j < label.size(); j++){
      Rect B = label[j];
      // if the boxes overlap
      if(rectOverlap(A, B)){
	// compute the area of A intersecate B
	int SI = areaOfIntersection(A,B);

	// compute the area of A union B
	int SU = A.width * A.height + B.width * B.height - SI;

	// compute ratio  [ 100% in case of perfect overlap, down to 0% ]
	float ratio = (float) SI / SU;

	if(ratio >= percentage)
	  checked[j]++;
      }
    }
  }
  
  int TP = 0, FP = 0, FN = 0; // true positives, false positives, false negatives;
  
  // compute TP and FN
  for(unsigned int i = 0; i < label.size(); i++){
    TP += checked[i];
    FN = checked[i] == 0 ? FN + 1 : FN;
  }
  
  // compute FP
  FP = detected.size() - TP;
 
  // return F1-score
  return (float) (2 * TP) / (2 * TP + FN + FP);
  
}

void violaAndCircle(Mat &original, Mat& viola_circle){
  // Detect dartboards and Display Result 
  Mat viola = original.clone();
  viola_circle = original.clone();
  std::vector<Rect> detectedObjects =  detectAndDisplay( viola );
  
  // HOUGH TRANSFORM TO REGIONS
  for(unsigned int i = 0; i < detectedObjects.size(); i++){
    int x = detectedObjects[i].x;
    int y = detectedObjects[i].y;
    int width = detectedObjects[i].width;
    int height = detectedObjects[i].height;
    
    //crop the region of interest (where the object is detected, i.e. where a rectangle is drawn)
    Mat cropedImage = original(Rect(x,y,width,height));
    
    // convert color
    Mat gray_image;
    cvtColor( cropedImage, gray_image, CV_BGR2GRAY );
    
    Mat xDerivative, yDerivative;
    // compute x-derivative and y-derivative image (same function, different kernels) 
    derivative(gray_image, xDerivative, dfdx);
    derivative(gray_image, yDerivative, dfdy);
    
    Mat magnitude;
    // compute the magnitude of the gradient
    gradientMagnitude(gray_image, magnitude, xDerivative, yDerivative);
    
    // compute the direction of the gradient
    Mat direction;
    gradientDirection(gray_image, direction, xDerivative, yDerivative);
    
    // apply Hough transform only on detected objects rather than the whole image
    //apply threshold
    Mat th_magnitude;
    int thr = TH_RING;
    threshold(magnitude, th_magnitude, thr);
    
    // apply hough transform
    Mat hough_image;
    houghTransform(th_magnitude, direction, thr, hough_image);
    
    // threshold the hough space
    Mat th_hough_space;
    threshold(hough_image, th_hough_space, thr);
    
    int count = 0;
    for(int ii = 0; ii < hough_image.rows; ii++)
      for(int jj = 0; jj < hough_image.cols; jj++){
	if(hough_image.at<uchar>(ii,jj) > THRESHOLD){
	  count++;
	}
      }
    
    // arbitrary threshold to draw a rectagle on the original image
    if(count > 10){
      rectangle(viola_circle, Point(x, y), Point(x + width, y + height), Scalar(0,255,0), 2);
    }
  }  
}


void hough_to_whole_image(Mat &frame, string imageName){
   //HOUGH TRANSFORM TO WHOLE IMAGE
  // convert color
  Mat gray_image;
  cvtColor( frame, gray_image, CV_BGR2GRAY );
  
  Mat xDerivative, yDerivative;
  // compute x-derivative and y-derivative image (same function, different kernels) 
  derivative(gray_image, xDerivative, dfdx);
  derivative(gray_image, yDerivative, dfdy);
  
  Mat magnitude;
  // compute the magnitude of the gradient
  gradientMagnitude(gray_image, magnitude, xDerivative, yDerivative);
  
  Mat direction;
  // compute the direction of the gradient
  gradientDirection(gray_image, direction, xDerivative, yDerivative);
  
  //apply threshold
  Mat th_magnitude;
  int thr = THRESHOLD;
  threshold(magnitude, th_magnitude, thr);
  
  // apply hough transform
  Mat hough_image;
  houghTransform(th_magnitude, direction, thr, hough_image);
  
  // threshold the hough space
  Mat th_hough_space;
  threshold(hough_image, th_hough_space, thr);
  
  // write results
  string imPath1 = "./task3/magnitude_" + imageName;
  string imPath2 = "./task3/hough_" + imageName;
  imwrite(imPath1, th_magnitude);
  imwrite(imPath2, hough_image);
}


vector<Vec4i> detectAndDisplayLine(Mat& image, Mat& th_magnitude, Mat& hough_line){
  vector<Vec4i> lines;
  // clone the original image
  hough_line = image.clone();
  
  // clone the magnitude (because HoughLinesP may modify the image)
  Mat clone_mag = th_magnitude.clone();
  /* detect lines using the thresholded magnitude.
     - 1 pixel for the radius, 1 degree for the angle, min votes, min line lenght, maxgap
  */
  HoughLinesP(clone_mag, lines, 1, CV_PI/180, 10, 20, 10 );
  // draw lines
  for( size_t i = 0; i < lines.size(); i++ ){
    Vec4i l = lines[i];
    line( hough_line, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(100,255,50), 1, CV_AA);
  }
  return lines;
}


void violaAndLine(Mat &original, Mat& viola_line){
  Mat viola = original.clone();
  viola_line = original.clone();
  std::vector<Rect> detectedObjects = detectAndDisplay( viola );

  for(unsigned int i = 0; i < detectedObjects.size(); i++){
    int x = detectedObjects[i].x;
    int y = detectedObjects[i].y;
    int width = detectedObjects[i].width;
    int height = detectedObjects[i].height;
    
    //crop the region of interest (where the object is detected, i.e. where a rectangle is drawn)
    Mat cropedImage = original(Rect(x,y,width,height));
    // convert color
    Mat gray_image;
    cvtColor( cropedImage, gray_image, CV_BGR2GRAY );

    Mat xDerivative, yDerivative;
    // compute x-derivative and y-derivative image (same function, different kernels) 
    derivative(gray_image, xDerivative, dfdx);
    derivative(gray_image, yDerivative, dfdy);
    
    // compute the magnitude of the gradient and threshold it
    Mat magnitude, th_magnitude;
    gradientMagnitude(gray_image, magnitude, xDerivative, yDerivative);
    threshold(magnitude, th_magnitude, THRESHOLD);
    
    //detect lines on the cropedImage
    //Mat hough_line;
    //vector<Vec4i> lines = detectAndDisplayLine(cropedImage, th_magnitude, hough_line);
    vector<Vec4i> lines;
    int minVotes = 20;
    int minLen = width/2;
    int maxGap = minLen/5;
    HoughLinesP(th_magnitude, lines, 1, CV_PI/180, minVotes, minLen, maxGap );

    int count = 0;
    if( lines.size() > 5)
      count = intersections(lines);
    if(count > 20){
      rectangle(viola_line, Point(x, y), Point(x + width, y + height), Scalar(0,255,0), 2);
    }
  }
  
}


int intersections(vector<Vec4i> lines){
  int count = 0;
  for(unsigned int i = 0; i < lines.size() - 1; i++){
    Point A(lines[i][0], lines[i][1]); // starting point of line 1
    Point B(lines[i][2], lines[i][3]); // end-point of line 1
    
    for(unsigned int j = i + 1; j < lines.size(); j++){
      Point C(lines[j][0], lines[j][1]); // starting point of line 2
      Point D(lines[j][2], lines[j][3]); // end-point of line 2
      Point E(B.x - A.x, B.y - A.y); // B - A, direction from A to B
      Point F(D.x - C.x, D.y - C.y); // D - C, direction from C to D
      
      Point P(-E.y, E.x); // point perpendicular to E
      
      int den = F.dot(P);
      // if den == 0 the lines are parallel
      if(den != 0) {
	// the lines may intersect
	float h = P.dot((A - C)) / den;
	// if h == 0 or h == 1 the segments touch at a end-point.
	// if h < 0 you have to go backward
	// if h > 1 you have to cross over the end point, so the segments do not intersect
	if( h >= 0 && h <= 1){
	  // the segments intersect
	  count++;
	}
      }
    }
  }
  return count;
}


vector<Rect> violaLineRing(Mat& frame, Mat& viola_line_ring){
  Mat viola = frame.clone();
  viola_line_ring = frame.clone();
  vector<Rect> detectedObjects = detectAndDisplay( viola );
  vector<Rect> comboObjs;

  for(unsigned int i = 0; i < detectedObjects.size(); i++){
    int x = detectedObjects[i].x;
    int y = detectedObjects[i].y;
    int width = detectedObjects[i].width;
    int height = detectedObjects[i].height;
    
    //crop the region of interest (where the object is detected, i.e. where a rectangle is drawn)
    Mat cropedImage = frame(Rect(x,y,width,height));
    // convert color
    Mat gray_image;
    cvtColor( cropedImage, gray_image, CV_BGR2GRAY );
    
    // compute x-derivative and y-derivative image (same function, different kernels) 
    Mat xDerivative, yDerivative;
    derivative(gray_image, xDerivative, dfdx);
    derivative(gray_image, yDerivative, dfdy);
    
    // compute the magnitude of the gradient and threshold it
    Mat magnitude, th_magnitude;
    gradientMagnitude(gray_image, magnitude, xDerivative, yDerivative);
    threshold(magnitude, th_magnitude, THRESHOLD);
    
    // compute the direction of the gradient
    Mat direction;
    gradientDirection(gray_image, direction, xDerivative, yDerivative);
    
    //detect lines on the cropedImage
    Mat hough_line;
    
    // apply hough transform to the region, to detect rings 
    Mat hough_image;
    houghTransform(th_magnitude, direction, THRESHOLD, hough_image);
    
    // compute the number of peaks
    int nVotes = 0;
    for(int ii = 0; ii < hough_image.rows; ii++)
      for(int jj = 0; jj < hough_image.cols; jj++)
	if(hough_image.at<uchar>(ii,jj) > THRESHOLD)
	  nVotes++;
    
    // if there are enough votes, there is probably a circle but not a dartboard
    if(nVotes >= 10){
      // lines detection
      vector<Vec4i> lines;
      int minVotes = 20;
      int minLen = width/2;
      int maxGap = minLen/5;
      HoughLinesP(th_magnitude, lines, 1, CV_PI/180, minVotes, minLen, maxGap);
      int nInter = 0;
      if( lines.size() > 5){
	// if there are enough lines, compute the number of intersections
	nInter = intersections(lines);
	if(nInter >= 25){
	  // if there are enough intersections, draw the box
	  rectangle(viola_line_ring, Point(x, y), Point(x + width, y + height), Scalar(0,255,0), 2);
	  // add the box to the detected objects
	  comboObjs.push_back(detectedObjects[i]);
	}
      }
    }
  }
  return comboObjs;
}


vector<Rect> improvedDetector(Mat& frame, Mat& detected){
  detected = frame.clone();
  vector<Rect> detectedObjects;
  vector<Rect> filteredBoxes;
  
  Mat frame_gray;
  // 1. Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  // 2. Perform Viola-Jones Object Detection 
  cascade.detectMultiScale( frame_gray, detectedObjects, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );  

  // for each region detected
  for(unsigned int i = 0; i < detectedObjects.size(); i++){
    // get coordinates of the box
    int x = detectedObjects[i].x;
    int y = detectedObjects[i].y;
    int width = detectedObjects[i].width;
    int height = detectedObjects[i].height;
    
    //crop the region of interest (where the object is detected, i.e. where a rectangle is drawn)
    Mat cropedImage = frame(Rect(x,y,width,height));
    // convert color
    Mat gray_image;
    cvtColor( cropedImage, gray_image, CV_BGR2GRAY );
    
    // compute x-derivative and y-derivative image (same function, different kernels) 
    Mat xDerivative, yDerivative;
    derivative(gray_image, xDerivative, dfdx);
    derivative(gray_image, yDerivative, dfdy);
    
    // compute the magnitude of the gradient and threshold it
    Mat magnitude, th_magnitude;
    gradientMagnitude(gray_image, magnitude, xDerivative, yDerivative);
    threshold(magnitude, th_magnitude, THRESHOLD);
    
    // compute the direction of the gradient 
    Mat direction; 
    gradientDirection(gray_image, direction, xDerivative, yDerivative);
    
    // apply hough transform to the region to detect rings 
    Mat hough_image;
    houghTransform(th_magnitude, direction, THRESHOLD, hough_image);
    
    // compute the number of peaks
    int nPeaks = 0;
    for(int ii = 0; ii < hough_image.rows; ii++)
      for(int jj = 0; jj < hough_image.cols; jj++)
	if(hough_image.at<uchar>(ii,jj) > 200)
	  nPeaks++;
    
    // if there are enough votes, there is probably a ring...
    if(nPeaks >= 5){
      // detect lines
      vector<Vec4i> lines;
      int minVotes = 20;
      int minLen = width/2;
      int maxGap = minLen/5;
      HoughLinesP(th_magnitude, lines, 1, CV_PI/180, minVotes, minLen, maxGap);
      int nInter = 0;
      if( lines.size() > 5){
	// if there are enough lines, compute the number of intersections
	nInter = intersections(lines);
	if(nInter >= 25){
	  // if there are enough intersections, compute histogram
	  int inx = width/3;         // inner x
	  int iny = height/3;        // inner y
	  int inWidth = width/2;     // inner width
	  int inHeight = height/2;   // inner height
	  Mat innerRegion = gray_image(Rect(inx, iny, inWidth, inHeight));
	  
	  // get difference between white values and black values
	  int diff = difference(innerRegion);
	  
	  if(diff < 5000){
	    rectangle(detected, Point(x, y), Point(x + width, y + height), Scalar(0,255,0), 2);
	    filteredBoxes.push_back(detectedObjects[i]);
	  }
	}
      }
    }
  }
  
  return filteredBoxes;

}


int difference(Mat& image){
  int black = 0;
  int white = 0;
  for(int i = 0; i < image.rows; i++)
    for(int j = 0; j < image.cols; j++)
      if(image.at<uchar>(i,j) > 128)
	white++;
      else
	black++;
  return abs(black - white);
} 

Mat lineDetection(Mat &original, Mat &th_magnitude){
  int nRho = sqrt((original.rows * original.rows) + (original.cols * original.cols)); 
  int nTheta = 90; 
  
  // make available a 2d array for the parameter space
  Mat hough(nRho, nTheta, CV_8UC1);
  
  
  // initialize hough space
  for(int i = 0; i < th_magnitude.rows; i++){
    for(int j = 0; j < th_magnitude.cols; j++){
      if(th_magnitude.at<uchar>(i,j) > THRESHOLD){
	// increment all elements on the curve roh = x * cos(theta) + y * sin(theta)
	for(int t = 0; t < nTheta; t++){
	  int Rho = j * cos(t) + i * sin(t);
	  if(Rho >= 0 && Rho < nRho)
	    hough.at<uchar>(Rho,t)++;
	}
      }
    }
  }
  return hough;
}


vector<Rect> initialize(string filename){
  int n; // number of objects

  FILE* f = fopen(filename.c_str(), "r");
  if( !f ){
    fprintf(stderr, "error reading file\n");
  }

  // read the number of objects
  fscanf(f, "%d", &n); 
  vector<Rect> label(n);
  // read the objects coordinates
  for(int i = 0; i < n; i++){
    fscanf(f, "%d %d %d %d", &(label[i].x), &(label[i].y), &(label[i].width), &(label[i].height));
  }
  
  return label;
}

// if a component of a rectangle is between the other rectangle
bool valueInRange(int value, int min, int max){ 
  return (value >= min) && (value <= max); 
}

// if rectangle A overlaps with B
bool rectOverlap(Rect A, Rect B){
    bool xOverlap = valueInRange(A.x, B.x, B.x + B.width) ||
                    valueInRange(B.x, A.x, A.x + A.width);

    bool yOverlap = valueInRange(A.y, B.y, B.y + B.height) ||
                    valueInRange(B.y, A.y, A.y + A.height);

    return xOverlap && yOverlap;
}

int areaOfIntersection(Rect A, Rect B){
  return max(0, min(A.x + A.width, B.x + B.width) - max(A.x, B.x)) * 
    max(0, min(A.y + A.height, B.y + B.height) - max(A.y, B.y) );
}

vector< vector<Rect> > splitInGroups(vector<Rect> detected){
  vector< vector<Rect> > groups;
  
  for(unsigned int i = 0; i < detected.size(); i++){
    bool inserted = false;
    Rect A = detected[i];

    // for each group
    for(unsigned int g = 0; g < groups.size(); g++){
      vector<Rect> gr = groups[g];
      // for each box in the group
      for(unsigned int j = 0; j < gr.size(); j++){
	Rect B = gr[j];
	// if they overlap
	if(rectOverlap(A, B)){
	  int SI = areaOfIntersection(A,B);
	  int SU = A.width * A.height + B.width * B.height - SI;
	  float ratio = (float) SI / SU;
	  // if the overlapping region is big enough
	  if(ratio >= 0.50){
	    gr.push_back(A);
	    inserted = true;
	    break;
	  }
	}
      }
      // if it's inserted in a group, stop visiting groups
      if(inserted)
	break;  
    }
    
    if(!inserted){
      // create a new group
      vector<Rect> ng;
      ng.push_back(A);
      groups.push_back(ng);
    }
    
  }
  
  return groups;
}

vector<Rect> averageBoxes(vector < vector<Rect> > groups){
  vector<Rect> averageBoxes;
  // for each group compute the average box
  for( unsigned int i = 0; i < groups.size(); i++){
    vector<Rect> g = groups[i];
    int avgx = 0;
    int avgy = 0;
    int avgwidth = 0;
    int avgheight = 0;
    // for each box in the group
    int size = (int)g.size();
    for(int j = 0; j < size; j++){
      avgx += g[j].x;
      avgy += g[j].y;
      avgwidth += g[j].width;
      avgheight += g[j].height;
    }
    averageBoxes.push_back(Rect(avgx/size, avgy/size, avgwidth/size, avgheight/size));
  }
  return averageBoxes;
}
