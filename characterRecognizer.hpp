#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

#ifdef _WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

class CharacterRecognizer {
	private:
		int classes;
		int samples;
		int size;
		int attributes;
		int totalSamples;
		int trainingSamples;
		int testingSamples;
		cv::Ptr<cv::ml::ANN_MLP> model;
		cv::Mat trainingSet;
		cv::Mat trainingSetClassifications;
		cv::Mat testingSet;
		cv::Mat testingSetClassifications;
		std::string getFilename(int, int);
		char getCharacter(int);
		void processDataset(std::string, cv::Mat&, cv::Mat&);
		void cropImage(cv::Mat&);
	public:
		CharacterRecognizer(int = 62, int = 500, int = 32);
		void processData(std::string = "data", std::string = "training.dat", std::string = "testing.dat");
		void trainModel(std::string = "training.dat");
		void saveModel(std::string = "ann_mlp.mdl");
		void loadModel(std::string = "ann_mlp.mdl");
		void testModel(std::string = "testing.dat");
		void predictText(std::string = "img/text.png");
};
