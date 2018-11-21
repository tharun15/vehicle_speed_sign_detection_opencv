#pragma once
#include <algorithm>
#include "Number.h"

class NumberDetector
{
private:
	// reckless from the method
	const vector<Point> ellipse;
	int ellipseWidth, ellipseHeight;
	Ptr<ml::ANN_MLP> ann;

	// for normal-canny
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy; // [Next, Previous, First_Child, Parent]
	Mat numberMask;

	// for dilated-canny
	vector< vector<Point> > contoursDilated;
	vector<Vec4i> hierarchyDilated; // [Next, Previous, First_Child, Parent]
	Mat numberMaskDilated;


	// finds a top-left point in the contour (@contour) and moves it slightly down and to the right (depending on @offset)
	Point findSeed(const vector<Point> &contour, int maxy)
	{
		Point bottomLeft;
		int mindistance = 1000;
		for (int i = 0; i != contour.size(); i++)
		{
			int calcuclatedDistance = contour[i].x + (maxy - contour[i].y);
			if (calcuclatedDistance < mindistance)
			{
				mindistance = calcuclatedDistance;
				bottomLeft = contour[i];
			}
		}

		return bottomLeft;
	}

	void moveSeed(Point &seed, const Mat &mask)
	{
		for (int i = 0; i != 10; i++)
		{
			if (mask.at<uchar>(seed.y, seed.x) < 20)
				break;

			seed.x += 1;
			seed.y -= 1;
		}
	}

	bool validateResult(const vector<Number> &numbers, int &speed)
	{
		bool ok = false;
		// do we have 1-3 counts?
		if (numbers.size() == 1)
		{
            cout << "speed is " << endl;
			speed = numbers[0].num;
			if (speed != 5)
			{
				//throw exception ("speed with one number other than 5");
			}
			ok = true;
		}
		else if (numbers.size() == 2)
		{
			speed = numbers[0].num * 10 + numbers[1].num;
			if (speed < 20 && speed != 15)
			{
				//throw exception ("speed with two counts below 20, but not 15");
			}
			else if (speed > 20 && speed % 10 != 0)
			{
				//throw exception ("speed with two stacks above 20, where the 2nd stack is not 0");
			}
			ok = true;
		}
		else if (numbers.size() == 3)
		{
			speed = numbers[0].num * 100 + numbers[1].num * 10 + numbers[2].num;

			if (numbers[2].num != 0)
			{
				//throw exception ("speed with three numbers at which the last is not 0!");
			}
			if (speed > 130)
			{
				//throw exception ("speed above 130 ... this will not be right :)");
			}
			ok = true;
		}
		else if (numbers.size() > 3)
		{
			string err = "weird sign ";
			for (int i = 0; i != numbers.size(); i++)
				err += numbers[i].num;
			//throw exception(err.c_str());
		}

		return ok;
	}

public:
	Mat transformMatrix; // Transformation matrix for alignment (signs)
	Mat transformMatrixDilated; // Transformation matrix for alignment (signs)
	int speed = -1; // identified speed from the character

	NumberDetector(const Ptr<ml::ANN_MLP> &ann, const vector<Point> &detectedEllipse, const Mat &edgeImg) : ann(ann), ellipse(detectedEllipse)
	{
		// creates a mask representing the outer part of an ellipse
		Rect ellipseRect = boundingRect(ellipse);
		Mat contourMask = Mat::zeros(ellipseRect.size(), CV_8U);
		contours.push_back(ellipse);
		drawContours(contourMask, contours, 0, Scalar(255), cv::FILLED, 8, noArray(), 100000, Point(-ellipseRect.x, -ellipseRect.y));
		bitwise_not(contourMask, contourMask);

		// remove the part that is outside the ellipse
		numberMask = edgeImg.clone()(ellipseRect); // this clone thing...
		morphologyEx(contourMask, contourMask, cv::MORPH_DILATE, Mat::ones(Size(5, 5), CV_8U));
		numberMask -= contourMask;

		// because sometimes the canny grins ... we are a little strengthening of the goods. This case with enhanced goods is treated separately (as a backup)
		morphologyEx(numberMask, numberMaskDilated, cv::MORPH_DILATE, Mat::ones(Size(3, 3), CV_8U));

		// Find contours in the picture with plain canny robbers
		contours.clear();
		findContours(numberMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

		// Find contours in the image with a dilated canny robie
		contoursDilated.clear();
		findContours(numberMaskDilated, contoursDilated, hierarchyDilated, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

		ellipseWidth = ellipseRect.width;
		ellipseHeight = ellipseRect.height;
	}

	// finds counts according to ordinary canny goods
	bool findNumbers(bool findDilatedNumbersEntryPoint = false)
	{
		if (!findDilatedNumbersEntryPoint)
			cout << "DETECTING NUMBERS (normal canny) " << endl;
		vector<Number> numbers;

		// find a map to settle the counts
		for (int i = 0; i != contours.size(); i++)
		{
			// Are not we on the outermost contour?
			if (hierarchy[i][Hierarchy::Parent] != -1)
				continue; // @SHADY ... no continue ?

			//filtering by size:
			Rect numberRect = boundingRect(contours[i]);
			if (numberRect.height < 20 || numberRect.height < ellipseHeight / 4 || numberRect.width * numberRect.height >(ellipseWidth * ellipseHeight) / 2)
				continue;

			RotatedRect rotatedNumberRect = minAreaRect(contours[i]);

			// is the rotation of more than 45 degrees? ... then weird things happen -> ignore
			if (fabs(rotatedNumberRect.angle) > 45)
				rotatedNumberRect.angle = 0;

			// find a transformation that will align the sign, and transform (sub) the picture with counts
			transformMatrix = getRotationMatrix2D(rotatedNumberRect.center, rotatedNumberRect.angle, 1);
			cv::warpAffine(numberMask, numberMask, transformMatrix, numberMask.size());

			break; // we just got the transformation ...
		}

		// Find odds
		for (int i = 0; i != contours.size() && transformMatrix.rows != 0; i++)
		{
			Number numberData;

			// skipping everything but the most outsiders
			if (hierarchy[i][Hierarchy::Parent] != -1)
				continue;

			// transform the contour
			transform(contours[i], contours[i], transformMatrix);

			// filtering by size:
			Rect numberRect = boundingRect(contours[i]);
			if (numberRect.height < 20 || numberRect.height < ellipseHeight / 4 || numberRect.width * numberRect.height > (ellipseWidth * ellipseHeight) / 2)
				continue;

			
			Mat numberMask_COPY = numberMask.clone();

			//search for a charging point (seed)
			Point seed = findSeed(contours[i], numberRect.height + numberRect.y); //we find a point within the contour
			moveSeed(seed, numberMask_COPY);

			// Strange seed?
			if (seed.x > numberMask_COPY.cols || seed.y > numberMask_COPY.rows)
				continue;

			// filling the contour of the count
			floodFill(numberMask_COPY, seed, Scalar(255), 0, Scalar(20), Scalar(100), cv::FLOODFILL_FIXED_RANGE);

			// trim
			if (numberRect.x + numberRect.width > numberMask_COPY.cols)
				numberRect.width = numberMask_COPY.cols - numberRect.x;
			if (numberRect.y + numberRect.height > numberMask_COPY.rows)
				numberRect.height = numberMask_COPY.rows - numberRect.y;

			Mat number = numberMask_COPY(numberRect);
			numberData.boundingRect = numberRect;

			// We separately check if it's one ... because it resets it to NN
			bool couldBeOne = false;
			if (numberData.boundingRect.width <= numberData.boundingRect.height * 0.3)
			{
				couldBeOne = true;
			}
			
			int res = -1;
			if (!couldBeOne)
			{
				// change the size to be suitable for entering the NN and creating an actual input pattern
				cv::resize(number, number, Size(imgWidthAndLength, imgWidthAndLength));
				Mat pattern(1, imgWidthAndLength * imgWidthAndLength, CV_32F);
				for (int r = 0; r != number.rows; r++)
				{
					for (int c = 0; c != number.cols; c++)
					{
						float val = (number.at<uchar>(r, c) > 20) ? 1.0 : 0.0;
						pattern.at<float>(0, r * number.rows + c) = val;
					}
				}

				// prediction with NN and saving the result
                return true;
				res = ann->predict(pattern);
			}
			else
			{
				res = 1;
			}
			
			numberData.num = res;
			numbers.push_back(numberData);
		}

		// Sort the X axis counts
		sort(numbers.begin(), numbers.end(), Number::numberDataComparator);

		bool ok = validateResult(numbers, speed);
		
		// Have you recognized it successfully?
		if (ok)
		{
			cout << "Detection successful, SPEED: " << speed << endl;
			return true;
		}

		return false;
	}

	// find counts according to reinforced canny goods ... do not call "findNumbers" after that!
	bool findDilatedNumbers()
	{
		cout << "DETECTING NUMBERS (dilated canny) " << endl;
		contours = contoursDilated;
		hierarchy = hierarchyDilated;
		numberMask = numberMaskDilated;

		return findNumbers(true);
	}
};
