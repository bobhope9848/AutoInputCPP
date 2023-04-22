#ifndef TESTING_H_EXISTS
#define TESTING_H_EXISTS

//#include "train.h"
#include <dlib/dnn.h>
#include <vector>
#include "include/resnet.h"
#include "frames.h"
using namespace dlib;

class testing
{
private:

	INPUT inputs[4];

protected:

public:
    static enum ValidInputs
		{
			left,
			right,
			up,
			down,
			start,
			select,
			a,
			b,
			begin,
			end
		};

	testing();

    testing(std::vector<frames> testingFrames);

    std::vector<ValidInputs> predict(std::string modelLoc);

	void emuHarness(std::vector<ValidInputs> inputs);

	void releaseCtrls();
















};

#endif