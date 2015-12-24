#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#ifdef _WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

class CharacterRecognizer {
	private:
		int classes, samples, size, attributes;
		int totalSamples, trainingSamples, testingSamples;
		int avgWidth, avgHeight;
		cv::Ptr<cv::ml::ANN_MLP> model;
		cv::Mat trainingSet, trainingSetClassifications;
		cv::Mat testingSet, testingSetClassifications;
		std::string getFilename(int, int);
		char getCharacter(int);
		void processDataset(std::string, cv::Mat&, cv::Mat&);
		void cropImage(cv::Mat&);
		std::vector<std::vector<cv::Point> > sortContours(std::vector<std::vector<cv::Point> >&, std::vector<cv::Vec4i>&);
	public:
		CharacterRecognizer(int = 62, int = 1016, int = 32);
		void processData(std::string = "data", std::string = "training", std::string = "testing");
		void trainModel(std::string = "training");
		void saveModel(std::string = "ann_mlp.mdl");
		void loadModel(std::string = "ann_mlp.mdl");
		void testModel(std::string = "testing");
		void predictText(std::string = "img/text.png");
};
