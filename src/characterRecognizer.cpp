#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "characterRecognizer.hpp"
#include "logger.hpp"

#ifdef _WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

CharacterRecognizer::CharacterRecognizer(int classes, int samples, int size) {
	Logger::log("Initializing CharacterRecognizer with parameters:");
	Logger::log("Classes = " + std::to_string(classes)
			+ ", Samples = " + std::to_string(samples)
			+ ", Size = " + std::to_string(size));

	this->classes = classes;
	this->samples = samples;
	this->size = size;
	attributes = size*size;
	totalSamples = classes*samples;
	trainingSamples = (int)(totalSamples*0.7); // 70% of the samples are used for training
	testingSamples = totalSamples-trainingSamples; // Remaining 30% for testing

	Logger::log("CharacterRecognizer initialized!");
}

std::string CharacterRecognizer::getFilename(int i, int j) {
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

char CharacterRecognizer::getCharacter(int value) {
	if (value < 10) // Digits
		return '0' + value;

	if (value < 36) // Upper-case characters
		return 'A' + value - 10;

	return 'a' + value - 36; // Lower-case characters
}

/*
 * Registers every pixel that is not white on the image
 * and uses the vector to create a bounding rectangle.
 * This gives us the cropped image
 */
void CharacterRecognizer::cropImage(cv::Mat& img) {
	std::vector<cv::Point> pixels;
	pixels.reserve(img.rows*img.cols);

	for (int j = 0; j < img.rows; ++j) {
		for (int i = 0; i < img.cols; ++i) {
			if (img.at<uchar>(j, i) != 255)
				pixels.push_back(cv::Point(i, j));
		}
	}

	cv::Rect crop = cv::boundingRect(pixels);
	if (crop.width > 3 && crop.height > 10)
		img = img(crop);
	else
		img = cv::Mat();
}

void CharacterRecognizer::processData(std::string path, std::string trainingFilePath, std::string testingFilePath) {
	// Default values
	if (path == "")
		path = "data";
	if (trainingFilePath == "")
		trainingFilePath = "training";
	if (testingFilePath == "")
		testingFilePath = "testing";

	Logger::log("Processing data from " + path + "/ to "
			+ trainingFilePath + ".dat and " + testingFilePath + ".dat ...");

	std::ofstream trainingFile(trainingFilePath + ".dat");
	std::ofstream testingFile(testingFilePath + ".dat");

	for (int i = 1; i <= classes; ++i) {
		for (int j = 1; j <= samples; ++j) {
			std::string imagePath = path + PATH_SEPARATOR + getFilename(i, j);
			cv::Mat img = cv::imread(imagePath, 0),
				tmp(size, size, CV_16U, cv::Scalar(0));
			if (img.type() != CV_8UC1)
				cv::cvtColor(img, img, CV_BGR2GRAY);
			cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
			cv::threshold(img, img, 50, 255, 0);
			cropImage(img);
			if (img.size() == cv::Size(0, 0)) {
				/*
				 * threshold() returns a blank image
				 * so cropImage() returns an empty image
				 */
				if (j <= (int)(samples*0.7))
					--trainingSamples;
				else
					--testingSamples;
				Logger::log("\tError occured while processing " + imagePath);
				continue;
			}
			resize(img, tmp, tmp.size());
			for (int x = 0; x < size; ++x) {
				for (int y = 0; y < size; ++y) {
					if (j <= (int)(samples*0.7))
						trainingFile << ((tmp.at<uchar>(x, y) == 255) ? 1 : 0);
					else
						testingFile << ((tmp.at<uchar>(x, y) == 255) ? 1 : 0);
				}
			}
			if (j <= (int)(samples*0.7))
				trainingFile << (i-1) << '\n';
			else
				testingFile << (i-1) << '\n';
		}
	}

	trainingFile.close();
	testingFile.close();

	std::ofstream trainingNum(trainingFilePath + ".num");
	std::ofstream testingNum(testingFilePath + ".num");

	trainingNum << trainingSamples << '\n';
	testingNum << testingSamples << '\n';

	trainingNum.close();
	testingNum.close();

	Logger::log("Data processed!");
}

int CharacterRecognizer::preprocessDataset(std::string path) {
	Logger::log("\tPreprocessing dataset from " + path + "...");

	std::ifstream file(path);
	int samplesNum;
	file >> samplesNum;

	Logger::log("\tDataset preprocessed!");

	return samplesNum;
}

void CharacterRecognizer::processDataset(std::string path, cv::Mat& data, cv::Mat& results) {
	Logger::log("\tProcessing dataset from " + path + "...");

	std::string line;
	std::ifstream file(path);

	if (file.is_open()) {
		int i = 0;
		while (getline(file, line)) {
			int j = 0;
			for (; j < attributes; j++)
				data.at<float>(i, j) = (line[j] == '1' ? 1.f : 0.f);
			results.at<float>(i++, std::stoi(line.substr(j))) = 1.f;
		}
	}
	file.close();

	Logger::log("\tDataset processed!");
}

void CharacterRecognizer::trainModel(std::string path) {
	if (path == "")
		path = "training"; // Default value

	Logger::log("Training model with data from " + path + ".dat ...");

	trainingSamples = preprocessDataset(path + ".num");
	trainingSet = cv::Mat(trainingSamples, attributes, CV_32F);
	trainingSetClassifications = cv::Mat(trainingSamples, classes, CV_32F);

	processDataset(path + ".dat", trainingSet, trainingSetClassifications);
	
	Logger::log("Training model...");

	cv::Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = attributes;
	layers.at<int>(1, 0) = 16;
	layers.at<int>(2, 0) = classes;

	model = cv::ml::ANN_MLP::create();

	model->setLayerSizes(layers);
	model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0.6, 1);
	model->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001));
	model->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.1, 0.1);

	cv::Ptr<cv::ml::TrainData> trainData =
		cv::ml::TrainData::create(trainingSet, cv::ml::ROW_SAMPLE, trainingSetClassifications);
	
	model->train(trainData);

	Logger::log("Model trained!");
}

void CharacterRecognizer::saveModel(std::string path) {
	if (path == "")
		path = "ann_mlp.mdl"; // Default value

	Logger::log("Saving model...");

	model->save(path);

	Logger::log("Model saved!");
}

void CharacterRecognizer::loadModel(std::string path) {
	if (path == "")
		path = "ann_mlp.mdl"; // Default value

	Logger::log("Loading model...");

	model = cv::ml::StatModel::load<cv::ml::ANN_MLP>(path);

	if (model.empty())
		Logger::log("Error loading model!");
	else
		Logger::log("Model loaded!");
}

void CharacterRecognizer::testModel(std::string path) {
	if (path == "")
		path = "testing"; // Default value

	Logger::log("Testing model with data from " + path + "...");

	testingSamples = preprocessDataset(path + ".num");
	testingSet = cv::Mat(testingSamples, attributes, CV_32F);
	testingSetClassifications = cv::Mat(testingSamples, classes, CV_32F);
	
	processDataset(path + ".dat", testingSet, testingSetClassifications);

	int correct = 0, wrong = 0, index, maxIndex, actualIndex;
	float value, maxValue;
	char predicted, actual;

	cv::Mat tmp, classificationResult(1, classes, CV_32F);
	for (int i = 0; i < testingSamples; ++i) {
		tmp = testingSet.row(i);
		model->predict(tmp, classificationResult);
		maxIndex = 0;
		value = 0.f;
		maxValue = classificationResult.at<float>(0, 0);
		for (index = 0; index < classes; ++index) {
			value = classificationResult.at<float>(0, index);
			if (value > maxValue) {
				maxValue = value;
				maxIndex = index;
			}
		}
		predicted = getCharacter(maxIndex);
		if (testingSetClassifications.at<float>(i, maxIndex) != 1.f) {
			for (actualIndex = 0; actualIndex < classes; ++actualIndex) {
				if (testingSetClassifications.at<float>(i, actualIndex) == 1.f) {
					actual = getCharacter(actualIndex);
					if (predicted == actual - 32 || predicted == actual + 32)
						++correct;
					else
						++wrong;
					break;
				}
			}
		} else {
			++correct;
		}
	}

	double correctRate = correct * 100.f / testingSamples;
	double wrongRate = wrong * 100.f / testingSamples;

	Logger::log("Model tested!");
	Logger::log("\tCorrect: " + std::to_string(correct)
			+ " (" + std::to_string(correctRate) + "%)");
	Logger::log("\tWrong: "	+ std::to_string(wrong)
			+ " (" + std::to_string(wrongRate) + "%)");
}

std::vector<std::vector<cv::Point> > CharacterRecognizer::sortContours(std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Vec4i>& hierarchy) {
	for (int i = 0; i < contours.size(); ++i) {
		if (hierarchy[i][3] == -1) { // Remove child contours
			contours.erase(contours.begin()+i);
			hierarchy.erase(hierarchy.begin()+i);
			--i;
		}
	}

	std::vector<std::vector<cv::Point> > ret;
	while (contours.size() > 0) {
		/*
		 * This algorithm put the letters in the correct order
		 * using their x and y coordinates
		 */
		int minY = INT_MAX;
		int minYHeight = 0;
		for (int i = 0; i < contours.size(); ++i) {
			cv::Rect rect = boundingRect(contours[i]);
			if (rect.y < minY) {
				minY = rect.y;
				minYHeight = rect.height;
			}
		}
		int index = 0;
		int minX = INT_MAX;
		for (int i = 0; i < contours.size(); ++i) {
			cv::Rect rect = boundingRect(contours[i]);
			if (rect.y < minY + minYHeight) {
				if (rect.x < minX) {
					minX = rect.x;
					index = i;
				}
			}
		}
		ret.push_back(contours[index]);
		contours.erase(contours.begin()+index);
	}

	return ret;
}

void CharacterRecognizer::predictText(std::string path) {
	if (path == "")
		path = "img/text.png"; // Default value

	cv::Mat img = cv::imread(path, 0), tmp,
		sample(1, attributes, CV_32F), scaled(size, size, CV_16U, cv::Scalar(0)),
		classificationResult(1, classes, CV_32F);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	int index, maxIndex;
	float value, maxValue;

	if (img.type() != CV_8UC1)
		cv::cvtColor(img, img, CV_BGR2GRAY);
	cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
	cv::threshold(img, img, 50, 255, 0);
	tmp = img.clone();

	cv::findContours(tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	std::vector<std::vector<cv::Point> > cleanContours = sortContours(contours, hierarchy);

	std::string predicted;
	int currentX = 0;
	int currentY = 0;
	int currentWidth = 0;
	int currentHeight = 0;
	for (int i = 0; i < cleanContours.size(); ++i) {
		cv::Rect rect = cv::boundingRect(cleanContours[i]);
		tmp = img(rect);
		cropImage(tmp);
		if (tmp.size() == cv::Size(0, 0))
			continue;
		resize(tmp, scaled, scaled.size());
		int j = 0;
		for (int x = 0; x < size; ++x)
			for (int y = 0; y < size; ++y)
				sample.at<float>(0, j++) = ((scaled.at<uchar>(x, y) == 255) ? 1 : 0);
		model->predict(sample, classificationResult);
		maxIndex = 0;
		maxValue = classificationResult.at<float>(0, 0);
		for (index = 0; index < classes; ++index) {
			value = classificationResult.at<float>(0, index);
			if (value > maxValue) {
				maxValue = value;
				maxIndex = index;
			}
		}

		if (rect.x > currentX + std::max(currentWidth, rect.width)*1.5 && i != 0)
			predicted += ' ';
		if (rect.y > currentY + currentHeight && i != 0)
			predicted += '\n';
		currentX = rect.x;
		currentY = rect.y;
		currentWidth = rect.width;
		currentHeight = rect.height;

		predicted += getCharacter(maxIndex);
	}

	Logger::log("Predicted text below:\n" + predicted);
}
