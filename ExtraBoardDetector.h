#pragma once
using namespace std;
using namespace cv;

class ExtraBoardDetector
{
private:
	// for vb methods
	const double maxSurfaceRatio = 0.3;

	// for normal-canny
	Rect searchArea;
	Mat edgeImg;
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int ellipseHeight;

	// for dilated-canny
	Rect searchAreaDilated;
	Mat edgeImgDilated;
	vector< vector<Point> > contoursDilated;
	vector<Vec4i> hierarchyDilated;


	// gets the lowest (max Y) point in the vector
	int getMaxYFromPoints(const vector<Point> &pts)
	{
		int y = 0;
		for (int i = 0; i != pts.size(); i++)
		{
			if (pts[i].y > y)
				y = pts[i].y;
		}

		return y;
	}

public:

	ExtraBoardDetector(vector<Point> ellipse, const Mat &edgeImage) : edgeImg(edgeImage)
	{
		//initialize the search area to the ellipse
		searchArea = boundingRect(ellipse);
		ellipseHeight = searchArea.height;

		// Move to the left and enlarge the width (to circumvent the sign after the width)
		searchArea.x -= searchArea.width / 3;
		searchArea.width += (int)(0.67 * searchArea.width);

		// trim x
		if (searchArea.x < 0)
			searchArea.x = 0;
		if (searchArea.width + searchArea.x > edgeImg.size().width)
			searchArea.width = edgeImg.size().width - searchArea.x;
		
		// move down (to skip the speed-limit) and increase the height (to approx. 3 characters)
		searchArea.y += searchArea.height;
		searchArea.height *= 3.14;

		// trim y
		if (searchArea.y > edgeImg.size().height)
			searchArea.y = edgeImg.size().height;
		if (searchArea.height + searchArea.y > edgeImg.size().height)
			searchArea.height = edgeImg.size().height - searchArea.y;

		edgeImg = edgeImg(searchArea); //Crop the image to ROI

		// diluting / strengthening canny goods ... this example is dealt with separately later
		searchAreaDilated = searchArea;
		cv::morphologyEx(edgeImg, edgeImgDilated, cv::MORPH_DILATE, Mat::ones(Size(3, 3), CV_8U));

		// find contours in both sub-images
		findContours(edgeImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // find the contours
		findContours(edgeImgDilated, contoursDilated, hierarchyDilated, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE); // find the contours
	}

	// find complementary signs with the help of ordinary Canny's edges
	void getBoundingBox(Mat const &rgb, Mat &result, bool dilatedEntryPoint = false)
	{
		int maxy = 0; // this will be the largest offset from the end of the speed limit sign
		for (int i = 0; i != contours.size(); i++)
		{
			
			if (!dilatedEntryPoint)
			{
				// Contour is a hole in the other?
				if (hierarchy[i][Hierarchy::Parent] != -1)
					continue;
			}
			else
			{
				// external contour?
				if (hierarchy[i][Hierarchy::Parent] == -1)
					continue;
			}

			// contours approximate with polygons
			double epsilon = 10.0;
			vector<Point> polygon;
			approxPolyDP(contours[i], polygon, epsilon, true);

			// we nailed a five-wheeler?
			if (polygon.size() == 4)
			{
				// calculate the contour and the approximate polygon
				double contourArea = cv::contourArea(contours[i]);
				double polyRectArea = minAreaRect(contours[i]).size.area();

				// Is not it enough like a rectangle (per plate)?
				if (fabs(polyRectArea / contourArea - 1) > maxSurfaceRatio)
					continue;

				// gets the lower point of the complementary sign
				int polyMaxY = getMaxYFromPoints(polygon);
				if (polyMaxY > maxy)
					maxy = polyMaxY;

				//cout << "contArea: " << contourArea << " | polyRectArea: " << polyRectArea << endl;
			}
		}

		// fixing the "bounding box" in the area of detected characters
		int extraOffset = ellipseHeight * 0.2;
		searchArea.height = maxy + ellipseHeight + 3 * extraOffset;
		searchArea.y -= ellipseHeight + extraOffset;

		result = rgb(searchArea);
	}

	// find complementary characters using dilated canny's edges ... do not call "getBoundingBox" after this!
	void getBoundingBoxDilated(const Mat &rgb, Mat &result)
	{
		searchArea = searchAreaDilated;
		edgeImg = edgeImgDilated;
		contours = contoursDilated;
		hierarchy = hierarchyDilated;

		getBoundingBox(rgb, result, true);
	}
};
