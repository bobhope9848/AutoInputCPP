#include "model.h"




Model::Model()
{

}

void Model::printNetwork(resnet::train_50 resnet50)
{
	std::cout << "Printing current network weights" << std::endl;
	std::cout << resnet50 << std::endl;

}