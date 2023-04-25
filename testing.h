#ifndef TESTING_H_EXISTS
#define TESTING_H_EXISTS

//#include "train.h"
#include <dlib/dnn.h>
#include <vector>
#include "include/resnet.h"
#include "frames.h"
#include "model.h"
using namespace dlib;

class testing : Model
{
private:

	INPUT inputs[4];
	INPUT ip;

protected:

public:



	std::string ValidInputs[10] = { "left", "right", "up", "down", "start", "select", "a", "b", "begin", "end" };

	testing(std::vector<frames> testingFrames);

	void predict(std::vector<frames> predictionFrames);

	void emuHarness(std::string input, bool releaseMode);


	//std::vector<rgb_pixel> grabScreen();

	void testing::printControls(std::string binaryCtrls);
















};

#endif