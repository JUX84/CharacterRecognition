#include "characterRecognizer.hpp"
#include "logger.hpp"


#include <iostream>
int main(int argc, char *argv[])
{
	if (argc == 1) {
		std::cout << "Usage (All arguments are optional):\n";
		std::cout << "\t-log [file] -- logs to file (default: cr.log)\n";
		std::cout << "\t-classes [1-62] -- number of classes (default: 62)\n";
		std::cout << "\t-samples [1-1016] -- number of samples by classes (default: 200)\n";
		std::cout << "\t-size [16-32] -- size of processed image (default: 32)\n";
		std::cout << "\t-save [file] -- save model to file (default: ann_mlp.mdl)\n";
		std::cout << "\t-load [file] -- load model from file (default: ann_mlp.mdl)\n";
		std::cout << "\t-predict [image] -- predict text in image (default: img/text.png)\n";
		std::cout << "\t-train [dataset] -- train from dataset (default: training(.num,.dat))\n";
		std::cout << "\t-test [dataset] -- test using dataset (default: testing(.num,.dat)\n";
		std::cout << "\t-process [folder] [train_dataset [test_dataset]]\n\t\t-- process images from folder to training and testing datasets\n";
		return 1;
	}
	std::vector<std::string> args = {"-log", "-classes", "-samples", "-size", "-train", "-save", "-test", "-load", "-predict", "-process"};
	std::vector<bool> argsb = {false, false, false, false, false, false, false, false, false, false};
	std::vector<std::string> targets = {"", "", "", "", "", "", "", "", "", "", "", ""};
	for (int i = 1; i < argc; ++i) {
		if (argv[i][0] == '-') {
			for (int j = 0; j < args.size(); ++j) {
				if (argv[i] == args[j]) {
					argsb[j] = true;
					if (i+1 < argc
							&& argv[i+1][0] != '-') {
						targets[j] = argv[++i];
						if (j == 9) {
							if (i+1 < argc
									&& argv[i+1][0] != '-')
								targets[j+1] = argv[++i];
							if (i+1 < argc
									&& argv[i+1][0] != '-')
								targets[j+2] = argv[++i];
						}
					}
				}
			}
		}
	}
	if (argsb[0])
		Logger::init(targets[0]);
	int classes = 62, samples = 200, size = 32; //default values
	/*if (argsb[1] && targets[1] != "") // Changing the number of classes leads to wrong results
		classes = std::stoi(targets[1]);*/
	if (argsb[2] && targets[2] != "")
		samples = std::stoi(targets[2]);
	if (argsb[3] && targets[3] != "") {
		int tmp = std::stoi(targets[3]);
		if (tmp == 16 || tmp == 32)
			size = tmp;
	}
	CharacterRecognizer CR(classes, samples, size);
	if (argsb[9])
		CR.processData(targets[9], targets[10], targets[11]);
	if (argsb[4])
		CR.trainModel(targets[4]);
	if (argsb[5])
		CR.saveModel(targets[5]);
	if (argsb[6])
		CR.testModel(targets[6]);
	if (argsb[7])
		CR.loadModel(targets[7]);
	if (argsb[8])
		CR.predictText(targets[8]);
	Logger::stop();
	return 0;
}
