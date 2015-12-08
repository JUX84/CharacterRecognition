#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

#ifdef _WIN32
	#define PATH_SEPARATOR "\\"
#else
	#define PATH_SEPARATOR "/"
#endif

#define SIZE 16
#define ATTRIBUTES SIZE*SIZE
#define CLASSES 62
#define SAMPLES 1016
#define TOTAL_SAMPLES CLASSES*SAMPLES
#define TRAINING_SAMPLES (int)(TOTAL_SAMPLES*0.7)//Number of samples in training dataset
#define TEST_SAMPLES TOTAL_SAMPLES-TRAINING_SAMPLES       //Number of samples in test dataset

int train_errors = 0, test_errors = 0;

/********************************************************************************
This function will read the csv files(training and test dataset) and convert them
into two matrices. classes matrix have 10 columns, one column for each class label. If the label of nth row in data matrix
is, lets say 5 then the value of classes[n][5] = 1.
********************************************************************************/
void readDataset(std::string filename, cv::Mat &data, cv::Mat &classes, int total_samples)
{
	std::cout << "Reading dataset " << filename << '\n';
	int label;
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen(filename.c_str(), "r");

	//read each row of the csv file
	for (int row = 0; row < total_samples; row++)
	{
		//for each attribute in the row
		for (int col = 0; col <= ATTRIBUTES; col++)
		{
			//if its the pixel value.
			if (col < ATTRIBUTES) {
				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(row, col) = pixelvalue;
			}//if its the label
			else if (col == ATTRIBUTES) {
				//make the value of label column in that row as 1.
				fscanf(inputfile, "%i", &label);
				classes.at<float>(row, label) = 1.0;

			}
		}
	}

	fclose(inputfile);
	std::cout << "Reading finished!\n";
}

void scaleDownImage(cv::Mat &originalImg, cv::Mat &scaledDownImage)
{
	for (int x = 0; x<SIZE; x++)
	{
		for (int y = 0; y<SIZE; y++)
		{
			int yd = ceil((float)(y*originalImg.cols / SIZE));
			int xd = ceil((float)(x*originalImg.rows / SIZE));
			scaledDownImage.at<uchar>(x, y) = originalImg.at<uchar>(xd, yd);
		}
	}
}

bool cropImage(cv::Mat &originalImage, cv::Mat &croppedImage)
{
	int row = originalImage.rows;
	int col = originalImage.cols;
	int tlx, tly, bry, brx;//t=top r=right b=bottom l=left
	tlx = tly = bry = brx = 0;
	float suml = 0;
	float sumr = 0;
	int flag = 0;

	/**************************top edge***********************/
	for (int x = 1; x<row; x++)
	{
		for (int y = 0; y<col; y++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				tly = x;
				break;
			}
		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}
	/*******************bottom edge***********************************/
	for (int x = row - 1; x>0; x--)
	{
		for (int y = 0; y<col; y++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				bry = x;
				break;
			}
		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}
	/*************************left edge*******************************/

	for (int y = 0; y<col; y++)
	{
		for (int x = 0; x<row; x++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				tlx = y;
				break;
			}
		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}

	/**********************right edge***********************************/

	for (int y = col - 1; y>0; y--)
	{
		for (int x = 0; x<row; x++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				brx = y;
				break;
			}
		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}
	int width = brx - tlx;
	int height = bry - tly;
	if (width == 0 || height == 0)
		return false;
	croppedImage = originalImage(cv::Rect(tlx, tly, width, height));
	return true;
}

void convertToPixelValueArray(cv::Mat &img, int pixelarray[])
{
	int i = 0;
	for (int x = 0; x<SIZE; x++)
	{
		for (int y = 0; y<SIZE; y++)
		{
			pixelarray[i] = (img.at<uchar>(x, y) == 255) ? 1 : 0;
			i++;

		}

	}
}

std::string getFilename(int i, int j) {
	return "img0"
		+ std::to_string(i / 10)
		+ std::to_string(i % 10)
		+ "-0"
		+ std::to_string(j / 1000)
		+ std::to_string(j % 1000 / 100)
		+ std::to_string(j % 100 / 10)
		+ std::to_string(j % 10)
		+ ".png";
}

void readData(std::string path, int samples_nb1, int samples_nb2) {
	std::cout << "Reading data from " << path << "/\n";
	ofstream training("training.dat");
	ofstream test("test.dat");
	for (int i = 1; i <= samples_nb1; ++i) {
		for (int j = 1; j <= samples_nb2; ++j) {
			std::string imagePath = path + PATH_SEPARATOR + getFilename(i, j);
			//std::cout << imagePath << std::endl;
			Mat img = imread(imagePath, 0);
			Mat output;
			blur(img, output, Size(5, 5));
			threshold(output, output, 50, 255, 0);
			Mat scaledDownImage(SIZE, SIZE, CV_8U, Scalar(0));
			int pixelValueArray[ATTRIBUTES];
			if (!cropImage(output, output)) {
				if (j <= (int)(samples_nb2*0.7))
					++train_errors;
				else
					++test_errors;
				continue;
			}
			scaleDownImage(output, scaledDownImage);
			convertToPixelValueArray(scaledDownImage, pixelValueArray);
			for (int k = 0; k < ATTRIBUTES; ++k) {
				if (j <= (int)(samples_nb2*0.7))
					training << pixelValueArray[k] << ",";
				else
					test << pixelValueArray[k] << ",";
			}
			if (j <= (int)(samples_nb2*0.7))
				training << (i - 1) << "\n";
			else
				test << (i - 1) << "\n";
		}
	}
	training.close();
	test.close();
	std::cout << "Finished reading!\n";
}

char get(int v) {
	if (v < 10)
		return '0' + v;
	if (v < 36)
		return 'A' + v - 10;
	return 'a' + v - 36;
}

void createAndTestMLP() {
	//matrix to hold the training sample
	cv::Mat training_set(TRAINING_SAMPLES-train_errors, ATTRIBUTES, CV_32F);
	//matrix to hold the labels of each taining sample
	cv::Mat training_set_classifications(TRAINING_SAMPLES-train_errors, CLASSES, CV_32F);
	//matric to hold the test samples
	cv::Mat test_set(TEST_SAMPLES-test_errors, ATTRIBUTES, CV_32F);
	//matrix to hold the test labels.
	cv::Mat test_set_classifications(TEST_SAMPLES-test_errors, CLASSES, CV_32F);

	//
	cv::Mat classificationResult(1, CLASSES, CV_32F);
	//load the training and test data sets.
	readDataset("training.dat", training_set, training_set_classifications, TRAINING_SAMPLES-train_errors);
	readDataset("test.dat", test_set, test_set_classifications, TEST_SAMPLES-test_errors);

	// define the structure for the neural network (MLP)
	// The neural network has 3 layers.
	// - one input node per attribute in a sample so 256 input nodes
	// - 16 hidden nodes
	// - 10 output node, one for each class.
	cv::Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = ATTRIBUTES;//input layer
	layers.at<int>(1, 0) = 16;//hidden layer
	layers.at<int>(2, 0) = CLASSES;//output layer

	Ptr<ANN_MLP> model = ANN_MLP::create();

	model->setLayerSizes(layers);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.6, 1);
	model->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001));
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);

	std::cout << "Creating training data\n";
	Ptr<TrainData> tdata = TrainData::create(training_set, ROW_SAMPLE, training_set_classifications);

	std::cout << "Training model\n";
	model->train(tdata);
	std::cout << "Training finished!\n";

	model->save("ann_mlp.mdl");

	// Test the generated model with the test samples.
	cv::Mat test_sample;
	//count of correct classifications
	int correct_class = 0;
	//count of wrong classifications
	int wrong_class = 0;

	// for each sample in the test set.
	for (int tsample = 0; tsample < TEST_SAMPLES-test_errors; tsample++) {
		// extract the sample

		test_sample = test_set.row(tsample);

		//try to predict its class

		model->predict(test_sample, classificationResult);
		/*The classification result matrix holds weightage  of each class.
		we take the class with the highest weightage as the resultant class */

		// find the class with maximum weightage.
		int maxIndex = 0;
		float value = 0.0f;
		float maxValue = classificationResult.at<float>(0, 0);
		for (int index = 1; index<CLASSES; index++)
		{
			value = classificationResult.at<float>(0, index);
			if (value>maxValue)
			{
				maxValue = value;
				maxIndex = index;
			}
		}

		std::cout << "Testing sample # " << tsample << " -> result: " << get(maxIndex);

		//Now compare the predicted class to the actural class. if the prediction is correct then\
				            //test_set_classifications[tsample][ maxIndex] should be 1.
//if the classification is wrong, note that.
		if (test_set_classifications.at<float>(tsample, maxIndex) != 1.0f)
		{
			// if they differ more than floating point error => wrong class
			wrong_class++;
			//find the actual label 'class_index'
			for (int class_index = 0; class_index<CLASSES; class_index++)
			{
				if (test_set_classifications.at<float>(tsample, class_index) == 1.0f)
				{
					std::cout << " instead of " << get(class_index);
					break;
				}
			}

		}
		else {
			// otherwise correct
			correct_class++;
		}
		std::cout << '\n';
	}

	printf("\nResults on the testing dataset\n"
		"\tCorrect classification: %d (%g%%)\n"
		"\tWrong classifications: %d (%g%%)\n",
		correct_class, (double)correct_class * 100 / (TEST_SAMPLES-test_errors),
		wrong_class, (double)wrong_class * 100 / (TEST_SAMPLES-test_errors));
}

void testMLP() {
	Ptr<ANN_MLP> model = StatModel::load<ANN_MLP>("ann_mlp.mdl");

	Mat img = imread("d.png", 0);
	Mat output;
	blur(img, output, Size(5, 5));
	threshold(output, output, 50, 255, 0);
	Mat scaledDownImage(SIZE, SIZE, CV_8U, Scalar(0));
	int pixelValueArray[ATTRIBUTES];
	Mat sample(1, ATTRIBUTES, CV_32F);
	if (cropImage(output, output)) {
		scaleDownImage(output, scaledDownImage);
		convertToPixelValueArray(scaledDownImage, pixelValueArray);
		for (int k = 0; k < ATTRIBUTES; ++k) {
			sample.at<float>(0, k) = pixelValueArray[k];
		}
		int result = model->predict(sample);
		std::cout << get(result) << std::endl;
	}
}

int main(int argc, char *argv[])
{
	readData("data", CLASSES, SAMPLES);
	createAndTestMLP();
	testMLP();
	while (1);
	return 0;
}
