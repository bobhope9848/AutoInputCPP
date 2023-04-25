#ifndef TRAIN_H
#define TRAIN_H


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include "include/resnet.h"
#include "frames.h"
#include "model.h"

using namespace dlib;
class train : Model
{

private:
protected:

public:
	
	
	train(std::vector<frames> sortedFrames);

	int trainer(std::vector<frames> sortedFrames);
	void regularLoad(int batchSize, dlib::rand& rnd, std::vector<frames>& sortedFrames, std::vector<matrix<rgb_pixel>>& images, std::vector<unsigned long>& labels);

	

};

#endif