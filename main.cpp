#include "characterRecognizer.hpp"

int main(int argc, char *argv[])
{
	CharacterRecognizer CR;
	CR.processData();
	CR.trainModel();
	CR.saveModel();
	CR.testModel();
	CR.loadModel();
	CR.predictText();
	return 0;
}
