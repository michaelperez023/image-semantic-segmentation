/**
 * @file main.cpp
 * Computer Vision Assignment 3
 * Dr. Abid
 * @author Michael Perez
 */

#include "opencv2/imgproc.hpp"
#include "MeanShift.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

// Global variables
const string image_name = "blocks_L-150x150.png";
Mat image, grayImage, editedImage, labImage, filteredImage;
int pixelsCount = 0;
int optimalThreshold1 = 0;
int optimalThreshold2 = 0;
int optimalThreshold3 = 0;

double maxBetweenVar = 0;

double w0 = 0;
double m0 = 0;
double c0 = 0;
double p0 = 0;

double w1 = 0;
double m1 = 0;
double c1 = 0;
double p1 = 0;

double w2 = 0;
double m2 = 0;
double c2 = 0;
double p2 = 0;
int histogram[256] = { 0 };
double c = 0;
double Mt = 0;
char key;

void MultilevelThresholding(cv::Mat& src)
{
	histogram[256] = { 0 };
	pixelsCount = src.cols * src.rows;

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			uchar value = src.at<uchar>(y, x);
			histogram[value]++;
		}
	}

	c = 0;
	Mt = 0;

	double p[256] = { 0 };
	for (int i = 0; i < 256; i++)
	{
		p[i] = (double)histogram[i] / (double)pixelsCount;
		Mt += i * p[i];
	}

	optimalThreshold1 = 0;
	optimalThreshold2 = 0;
	optimalThreshold3 = 0;

	maxBetweenVar = 0;

	w0 = 0;
	m0 = 0;
	c0 = 0;
	p0 = 0;

	w1 = 0;
	m1 = 0;
	c1 = 0;
	p1 = 0;

	w2 = 0;
	m2 = 0;
	c2 = 0;
	p2 = 0;
	for (int tr1 = 0; tr1 < 256; tr1++)
	{
		p0 += p[tr1];
		w0 += (tr1 * p[tr1]);
		if (p0 != 0)
		{
			m0 = w0 / p0;
		}

		c0 = p0 * (m0 - Mt) * (m0 - Mt);

		c1 = 0;
		w1 = 0;
		m1 = 0;
		p1 = 0;
		for (int tr2 = tr1 + 1; tr2 < 256; tr2++)
		{

			p1 += p[tr2];
			w1 += (tr2 * p[tr2]);
			if (p1 != 0)
			{
				m1 = w1 / p1;
			}

			c1 = p1 * (m1 - Mt) * (m1 - Mt);


			c2 = 0;
			w2 = 0;
			m2 = 0;
			p2 = 0;
			for (int tr3 = tr2 + 1; tr3 < 256; tr3++)
			{

				p2 += p[tr3];
				w2 += (tr3 * p[tr3]);
				if (p2 != 0)
				{
					m2 = w2 / p2;
				}

				c2 = p2 * (m2 - Mt) * (m2 - Mt);

				c = c0 + c1 + c2;

				if (maxBetweenVar < c)
				{
					maxBetweenVar = c;
					optimalThreshold1 = tr1;
					optimalThreshold2 = tr2;
					optimalThreshold3 = tr3;
				}
			}
		}
	}
}

/**
 * @function main
 */
int main(int argc, char** argv)
{
	// Display windows
	image = imread(image_name);
	
	imshow("Image", image);

	if (image.empty())
	{
		cout << "The image could not be read. Exiting Program. ";
		return -1;
	}

	// Main loop
	while (1)
	{
		key = waitKey(10); // Scan for key press every 10 ms

		if (key == '1') // Apply Otsu Binarization method with two classes
		{
			if (image.channels() == 1)
			{
				threshold(image, editedImage, 0, 255, THRESH_OTSU);
				imshow("Edited Image", editedImage);
			}
			else if (image.channels() == 3)
			{
				Mat bgr[3];   //destination array
				Mat bgr_otsu[3];   //destination array
				vector<Mat> array_to_merge;

				split(image, bgr); // Split iamge

				threshold(bgr[0], bgr_otsu[0], 0, 255, THRESH_OTSU);
				threshold(bgr[1], bgr_otsu[1], 0, 255, THRESH_OTSU);
				threshold(bgr[2], bgr_otsu[2], 0, 255, THRESH_OTSU);

				array_to_merge.push_back(bgr_otsu[0]);
				array_to_merge.push_back(bgr_otsu[1]);
				array_to_merge.push_back(bgr_otsu[2]);

				merge(array_to_merge, editedImage); // Merge images

				imshow("Edited Image", editedImage);
			}
		}

		if (key == '2') // Apply Otsu Binarization method with multiple classes
		{
			// I first convert the image to grayscale
			cvtColor(image, grayImage, COLOR_BGR2GRAY);

			// Assign image for easy assigning of pixel values later
			editedImage = grayImage;

			// Apply the thresholding algorithm above
			MultilevelThresholding(grayImage);

			// Output the optimal pixel threshold values the algorithm outputs
			cout << "Optimal Threshold 1: " << optimalThreshold1 << endl;
			cout << "Optimal Threshold 2: " << optimalThreshold2 << endl;
			cout << "Optimal Threshold 3: " << optimalThreshold3 << endl;
			
			Scalar intensity = 0;

			// Segment image based on the three thresholds
			for (int x = 0; x < grayImage.rows; x++)
			{
				for (int y = 0; y < grayImage.cols; y++)
				{
					intensity = grayImage.at<uchar>(x, y);
					//cout << "X: " << x << ", Y: " << y << ", value: " << intensity << endl;
					if (intensity.val[0] < optimalThreshold1)
						editedImage.at<uchar>(x, y) = 31;
					else if (intensity.val[0] >= optimalThreshold1 &&  intensity.val[0] < optimalThreshold2)
						editedImage.at<uchar>(x, y) = 93;
					else if (intensity.val[0] >= optimalThreshold2 && intensity.val[0] < optimalThreshold3)
						editedImage.at<uchar>(x, y) = 156;
					else if (intensity.val[0] >= optimalThreshold3)
						editedImage.at<uchar>(x, y) = 218;
				}
			}
		
			imshow("Edited Image", editedImage);
		}

		if (key == '3') // Apply Mean Shift segmentation to the image
		{
			// Convert color from RGB to Lab
			cvtColor(image, labImage, COLOR_RGB2Lab);

			// Initilize Mean Shift with spatial bandwith and color bandwith
			MeanShift MSProc(8, 16);

			// Filtering Process
			MSProc.MSFiltering(labImage);
			filteredImage = labImage;

			// Convert color from Lab to RGB
			cvtColor(filteredImage, editedImage, COLOR_Lab2RGB);

			imshow("Filtered Image", editedImage);

			// Segmentation Process include Filtering Process (Region Growing)
			MSProc.MSSegmentation(labImage);

			// Print the bandwith
			cout << "the Spatial Bandwith is " << MSProc.hs << endl;
			cout << "the Color Bandwith is " << MSProc.hr << endl;

			// Convert color from Lab to RGB
			cvtColor(labImage, editedImage, COLOR_Lab2RGB);

			imshow("Edited Image", editedImage);
		}
	}
	return 0;
}
