#ifndef FRAMES_H
#define FRAMES_H
#include <bitset>


struct frames
{
	std::string frameName;
	unsigned long controls;

	frames(std::string frame, unsigned long ctrls)
	{
		this->frameName = frame;
		this->controls = ctrls;
	
	}

	//Returns vector
	std::vector<int> controlsVector()
	{
		std::string binaryeqv = std::bitset<8>((controls)).to_string();

		std::vector<int> binaryVector;
		for (int i = 0; i < binaryeqv.length(); i++)
		{
			//Subtract ascii value of '0' from ascii value of binaryeqv[i]
			binaryVector.push_back((binaryeqv[i]) - '0');
		}

		return binaryVector;
	}
};

#endif