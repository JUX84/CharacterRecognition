#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::ml;

#ifdef _WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

#define SIZE 32											// Size (Width & Height) of image
#define ATTRIBUTES SIZE*SIZE							// Number of attributes (pixels)
#define CLASSES 62										// Number of classes (26 letters -upper/lower- and 10 digits)
#define SAMPLES 200										// Number of sample by class
#define TOTAL_SAMPLES CLASSES*SAMPLES					// Total number of samples
#define TRAINING_SAMPLES (int)(TOTAL_SAMPLES*0.7)		// Number of samples in training dataset (70% of all data)
#define TEST_SAMPLES TOTAL_SAMPLES-TRAINING_SAMPLES		// Number of samples in test dataset (30% ...)

int train_errors = 0, test_errors = 0;

void readDataset(std::string filename, Mat &data, Mat &classes, int total_samples)
{
	std::cout << "Reading dataset " << filename << '\n';
	int label;
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen(filename.c_str(), "r");

	//read each row of the file
	for (int row = 0; row < total_samples; row++)
	{
		//for each attribute in the row
		for (int col = 0; col <= ATTRIBUTES; col++)
		{
			if (col < ATTRIBUTES) { // pixel value
				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(row, col) = pixelvalue;
			} else if (col == ATTRIBUTES) { // the label
				fscanf(inputfile, "%i", &label);
				classes.at<float>(row, label) = 1.0;
			}
		}
	}
	fclose(inputfile);
	std::cout << "Reading finished!\n";
}

bool cropImage(Mat &input, Mat &output) {
	// vector with all non-black point positions
	std::vector<cv::Point> pixels;
	pixels.reserve(input.rows*input.cols);

	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {
			if (input.at<uchar>(j, i) != 255)
				pixels.push_back(cv::Point(i,j));
		}
	}

	// create bounding rect around those points
	cv::Rect crop = cv::boundingRect(pixels);

	output = input(crop);
	return (crop.width != 0 && crop.height != 0);
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
	std::ofstream training("training.dat");
	std::ofstream test("test.dat");
	for (int i = 1; i <= samples_nb1; ++i) {
		for (int j = 1; j <= samples_nb2; ++j) {
			std::string imagePath = path + PATH_SEPARATOR + getFilename(i, j);
			//std::cout << imagePath << std::endl;
			Mat img = imread(imagePath, 0);
			Mat output;
			GaussianBlur(img, output, Size(5, 5), 0);
			threshold(output, output, 50, 255, 0);
			if (!cropImage(output, output)) {
				if (j <= (int)(samples_nb2*0.7))
					++train_errors;
				else
					++test_errors;
				std::cout << "Error while processing " << imagePath << std::endl;
				continue;
			}
			Mat scaledDownImage(SIZE, SIZE, CV_16U, Scalar(0));
			resize(output, scaledDownImage, scaledDownImage.size());
			for (int x = 0; x<SIZE; x++)
			{
				for (int y = 0; y<SIZE; y++)
				{
					if (j <= (int)(samples_nb2*0.7))
						training << ((scaledDownImage.at<uchar>(x, y) == 255) ? 1 : 0) << ",";
					else
						test << ((scaledDownImage.at<uchar>(x, y) == 255) ? 1 : 0) << ",";
				}
			}
			if (j <= (int)(samples_nb2*0.7))
				training << (i - 1) << "\n";
			else
				test << (i - 1) << "\n";
		}
	}
	training.close();
	test.close();
	std::cout << "Reading finished!\n";
}

char get(int v) {
	if (v < 10)
		return '0' + v;
	if (v < 36)
		return 'A' + v - 10;
	return 'a' + v - 36;
}

void createAndTestMLP() {
	Mat training_set(TRAINING_SAMPLES-train_errors, ATTRIBUTES, CV_32F);
	Mat training_set_classifications(TRAINING_SAMPLES-train_errors, CLASSES, CV_32F);
	
	Mat test_set(TEST_SAMPLES-test_errors, ATTRIBUTES, CV_32F);
	Mat test_set_classifications(TEST_SAMPLES-test_errors, CLASSES, CV_32F);

	Mat classificationResult(1, CLASSES, CV_32F);
	//load the training and test data sets.
	readDataset("training.dat", training_set, training_set_classifications, TRAINING_SAMPLES-train_errors);
	readDataset("test.dat", test_set, test_set_classifications, TEST_SAMPLES-test_errors);

	// define the structure for the neural network (MLP)
	// The neural network has 3 layers.
	// - one input node per attribute in a sample so 256 input nodes
	// - 16 hidden nodes
	// - 10 output node, one for each class.
	Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = ATTRIBUTES;//input layer
	layers.at<int>(1, 0) = 16;//hidden layer
	layers.at<int>(2, 0) = CLASSES;//output layer

	Ptr<ANN_MLP> model = ANN_MLP::create();

	model->setLayerSizes(layers);
	model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.6, 1);
	model->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001));
	model->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);

	Ptr<TrainData> tdata = TrainData::create(training_set, ROW_SAMPLE, training_set_classifications);

	std::cout << "Training model\n";
	model->train(tdata);
	std::cout << "Training finished!\n";

	model->save("ann_mlp.mdl");

	// Test the generated model with the test samples.
	Mat test_sample;
	//count of correct classifications
	int correct = 0;
	//count of wrong classifications
	int wrong = 0;

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

		char c = get(maxIndex);

		std::cout << "Testing sample # " << tsample << " -> result: " << c;

		//Now compare the predicted class to the actural class. if the prediction is correct then\
		//test_set_classifications[tsample][ maxIndex] should be 1.
		//if the classification is wrong, note that.
		if (test_set_classifications.at<float>(tsample, maxIndex) != 1.0f)
		{
			//find the actual label 'class_index'
			int class_index = 0;
			for (; class_index<CLASSES; class_index++)
			{
				if (test_set_classifications.at<float>(tsample, class_index) == 1.0f)
				{
					char tmp = get(class_index);
					std::cout << " instead of " << tmp;
					if (c == tmp - 32 || c == tmp + 32) {
						std::cout << " - OK !";
						correct++;
					} else {
						// if they differ more than floating point error => wrong class
						wrong++;
					}
					break;
				}
			}
		} else {
			// otherwise correct
			correct++;
		}
		std::cout << '\n';
	}

	double correct_rate = correct * 100.0 / (TEST_SAMPLES-test_errors);
	double wrong_rate = wrong * 100.0 / (TEST_SAMPLES-test_errors);
	std::cout << "\nResults:";
	std::cout << "\n\tCorrect: " << correct << " (";
	std::cout << correct_rate << "%)";
	std::cout << "\n\tWrong: " << wrong << " (";
	std::cout << wrong_rate << "%)\n\n";
}

void testMLP(std::string path) {
	Ptr<ANN_MLP> model = Algorithm::load<ANN_MLP>("ann_mlp.mdl");

	Mat original = imread(path, 0), scaledDown(SIZE, SIZE, CV_16U, Scalar(0)), sample(1, ATTRIBUTES, CV_32F), tmp;
	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	if (original.type() != CV_8UC1)
		cvtColor(original, tmp, CV_BGR2GRAY);
	GaussianBlur(original, tmp, Size(5, 5), 0);
	threshold(tmp, tmp, 50, 255, 0);
	imshow("Contours", tmp);
	waitKey();
	findContours(tmp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat classificationResult(1, CLASSES, CV_32F);
	for (int i = 0; i < contours.size(); ++i) {
		if (contourArea(contours[i]) > 100) {
			tmp = original(boundingRect(contours[i]));
			resize(tmp, scaledDown, scaledDown.size());
			int j = 0;
			for (int x = 0; x<SIZE; x++)
			{
				for (int y = 0; y<SIZE; y++)
				{
					sample.at<float>(0, j++) = ((scaledDown.at<uchar>(x, y) == 255) ? 1 : 0);
				}
			}
			model->predict(sample, classificationResult);
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
			std::cout << "Prediction: " << get(maxIndex) << std::endl;
			imshow("Contours", tmp);
			waitKey();
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc > 1) {
		testMLP(argv[1]);
	} else {
		readData("data", CLASSES, SAMPLES);
		createAndTestMLP();
	}
	return 0;
}
