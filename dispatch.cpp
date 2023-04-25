#include <iostream>
#include <vector>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/misc_api.h>
#include <string>
#include <sstream>
#include <bitset>
#include <array>
#include "frames.h"
#include "train.h"
#include "testing.h"




//Prototypes
std::vector<frames> loadData(std::string dirPath);
unsigned long translateCtrls(std::string fileName);
std::vector<std::string> split(std::string ssplit, char c);
bool compareFileNum(dlib::file file1, dlib::file file2);


//Globals
std::vector<frames> inputDataset = loadData("C:/Users/bobhope/Documents/College Work/CSCI 24000/AutoInputCPP/data/train");
std::vector<frames> validationSet = loadData("C:/Users/bobhope/Documents/College Work/CSCI 24000/AutoInputCPP/data/validate");

int main()
{
	std::cout << "Initilizing" << std::endl;
	train trainer(inputDataset);
	testing tester(validationSet);
	return 0;
}


std::vector<frames> loadData(std::string dirPath)
{
	std::vector<frames> imgs;


	std::vector<dlib::directory> levels = dlib::directory(dirPath).get_dirs();
	std::vector<dlib::file> files;
	//Get each level
	for (dlib::directory level : levels)
	{
		//Get each session
		for (dlib::directory session : level.get_dirs())
		{
			//Get all files
			files.clear();
			files = session.get_files();
			//std::sort(files.begin(), files.end(), compareFileNum);
			for (dlib::file img : files)
			{
				//Calls translateCtrls and returns vector float array.
				unsigned long ctrls = translateCtrls(img.name());

				//Feeds frame name and vector ctrls into imgs vector.
				imgs.push_back(frames(img.full_name(), ctrls));
			}
		
		}

		std::cout << level.name() + " has completed succesfully" << std::endl;

	}
	return imgs;
}



unsigned long translateCtrls(std::string fileName)
{
	std::string holder;
	std::vector<std::string> substringArray = split(fileName, 'a');

	holder = substringArray[1];
	holder.erase(holder.end() - 4, holder.end());
	//std::string binaryeqv = std::bitset<8>(std::stoi(holder)).to_string();
	std::string binaryeqv = holder;

	return std::stoul(binaryeqv);

}

#pragma region Support_Methods

std::vector<std::string> split(std::string ssplit, char c)
{
	std::stringstream ss;
	ss.str(ssplit);
	std::string holder;
	std::vector<std::string> substringVector;
	if (ssplit.find(c) != std::string::npos)
	{
		for(int i = 0 ; getline(ss, holder, c); i++)
		{
			substringVector.push_back(holder);
		}
	}
	else
	{
		throw "Delimiter not found in string";
	}
	
	return substringVector;

}

//Custom comparison operator for files (Python = 1 line)
//NOTE: Can't handle full paths, just filenames
//filename example: "1_a20.png"
bool compareFileNum(dlib::file file1, dlib::file file2)
{
	std::vector<std::string> file1vec = split(file1.name(), '_');
	std::vector<std::string> file2vec = split(file2.name(), '_');
	
	return(std::stoi(file1vec[0]) < std::stoi(file2vec[0]));


}

#pragma endregion