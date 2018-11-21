#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/opencv.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <dirent.h>
#include <unistd.h>
#include <glob.h>
#define GetCurrentDir getcwd
#define imgWidthAndLength 28
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "Hierarchy.h"
#include "ExtraBoardDetector.h"
#include "NumberDetector.h"
#include "EllipseDetector.h"
#include "NN.h"


using namespace cv;
using namespace std;

void equalizeImage(const Mat &bgr, Mat &equalized)
{
    vector<Mat> channels;
    cvtColor(bgr, equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
    split(equalized, channels); //split the image into channels
    equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
    merge(channels, equalized); //merge 3 channels including the modified 1st channel into one image
    cvtColor(equalized, equalized, CV_YCrCb2BGR);
}

void resizeImage(Mat &in)
{
    int width = in.cols;
    int height = in.rows;
    const int maxDim = 1000;
    
    // Scale the biggest page on @maxDim
    if (width > height && width > maxDim)
    {
        double scale = (double)maxDim / (double)width;
        resize(in, in, Size(maxDim, height * scale));
    }
    else if (height > width && height > maxDim)
    {
        double scale = (double)maxDim / (double)height;
        resize(in, in, Size(width * scale, maxDim));
    }
}

vector<Mat> detectSpeedSign(Ptr<ml::ANN_MLP> ann, Mat &imgBGR, Mat &img_gray)
{
    vector<Mat> detectedSpeedSigns;
    Mat equalizedBGRimg = imgBGR;
    //equalizeImage(imgBGR, equalizedBGRimg); // Improve contrast in the image
    
    // edge detection
    int cannyThresh = 50;
    Mat edgeImg;
    
    Canny(img_gray, edgeImg, cannyThresh, cannyThresh*3);
    imshow("edge image",edgeImg);
    
    // contour search
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(edgeImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    // detection of elips
    vector< vector<Point> > ellipses;
    EllipseDetector ellipseDetector(contours, hierarchy);
    ellipseDetector.detectEllipses(); // Find an ellipse in the picture
    ellipses = ellipseDetector.ellipses;

    for (int i = 0; i != ellipses.size(); i++)
    {
        bool success = false;
        
        // Find the digits in the character
        NumberDetector numberDetector(ann, ellipses[i], edgeImg);
        try
        {
            success = numberDetector.findNumbers();
        }
        catch (exception e)
        {
            cerr << "NUMBERDETECTOR EXCEPTION: " << e.what() << endl;
        }
        
        try
        {
            if (!success)
                success = numberDetector.findDilatedNumbers();
        }
        catch (exception e)
        {
            cerr << "NUMBERDETECTOR EXCEPTION: " << e.what() << endl;
        }
        
        Mat resultImage; // there will be a (potential) result
        if (success)
        {
            
            // find complementary characters
            Mat resultImage1, resultImage2;
            ExtraBoardDetector extraBoardDetector(ellipses[i], edgeImg);
            extraBoardDetector.getBoundingBox(equalizedBGRimg, resultImage1);
            extraBoardDetector.getBoundingBoxDilated(equalizedBGRimg, resultImage2);
            
            // select the best result and align the result image
            if (resultImage1.size().height >= resultImage2.size().height)
            {
                resultImage = resultImage1;
            }
            else
            {
                resultImage = resultImage2;
            }
            
            if (!numberDetector.transformMatrix.empty())
                warpAffine(resultImage, resultImage, numberDetector.transformMatrix, resultImage.size(), cv::INTER_LINEAR, cv::BorderTypes::BORDER_REPLICATE);
            
            detectedSpeedSigns.push_back(resultImage);
        }
    }

    return detectedSpeedSigns;
}

void processImage(Mat &imgBGR, Ptr<ml::ANN_MLP> ann, int imgDisplayTime)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // converted to grayscale
    Mat imgGrey;
    cvtColor(imgBGR, imgGrey, cv::COLOR_BGR2GRAY);
    
    vector<Mat> speedSigns = detectSpeedSign(ann, imgBGR, imgGrey);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << ms.count() << "ms" << endl;
    
    // show characters
    string tag = "Detected Speed Sign";
    cout << speedSigns.size() << endl;


    for (int i = 0; i != speedSigns.size(); i++)
    {
        cout << "in for loop" << endl;
        cout << speedSigns.size() << endl;
        namedWindow(tag, cv::WINDOW_AUTOSIZE);
        imshow(tag, speedSigns[i]);
        moveWindow(tag, 21, 21);
        
        waitKey(0);
    }
}

std::string GetCurrentWorkingDir( void ) {
    char buff[FILENAME_MAX];
    GetCurrentDir( buff, FILENAME_MAX );
    std::string current_working_dir(buff);
    return current_working_dir;
}

int main(int argc, char** argv)
{
    bool READ_VIDEO = true;
    string VIDEO_PATH = "17.mp4";
    string IMG_PATH = "images_test/5.jpg";
    Mat imgBGR;
    std::cout << "i am now in " << GetCurrentWorkingDir() << endl;
    Ptr<ml::ANN_MLP> ann = loadANN();
    
    if (READ_VIDEO)
    {
        cout << VIDEO_PATH;
        cv::VideoCapture video(VIDEO_PATH);
        while(true)
        {
            video >> imgBGR;
            resizeImage(imgBGR);
            
            if (!imgBGR.empty())
            {
                // I will display the current original image
                namedWindow("Video", cv::WINDOW_AUTOSIZE);
                imshow("Video", imgBGR);
                waitKey(18);
                // Process the image
                processImage(imgBGR, ann, 18);
            }
            else
                break;
        }
    }
    else
    {
        vector<cv::String> fn;
        vector<cv::Mat> data;
        cv::String path(IMG_PATH);
        cv::glob(path,fn,true); // recurse
        for (size_t k=0; k<fn.size(); ++k)
        {
            cout <<  fn.size() << std::endl ;
            imgBGR = imread(fn[k], IMREAD_COLOR);
            if( imgBGR.empty() )                      // Check for invalid input
            {
                cout <<  "Could not open or find the image" << std::endl ;
                return -1;
            }
            imshow("Original Image", imgBGR);
            if (imgBGR.empty()) continue; //only proceed if sucsessful
            //waitKey(18);
            processImage(imgBGR, ann, 5000);
            data.push_back(imgBGR);
        }
    }
    
    return 0;
}
