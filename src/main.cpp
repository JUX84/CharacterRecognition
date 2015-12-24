#include "characterRecognizer.hpp"
#include "logger.hpp"

int main(int argc, char *argv[])
{
	Logger::init();
	CharacterRecognizer CR;
	/*CR.processData();
	CR.trainModel();
	CR.saveModel();
	CR.testModel();*/
	CR.loadModel();
	CR.predictText();
	Logger::stop();
	return 0;
}
