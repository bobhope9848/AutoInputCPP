#include "testing.h"
#include <vector>
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <Windows.h>
#include <array>
#include <bitset>

/*
testing::testing()
{
	
    //grabScreen();

}
*/

testing::testing(std::vector<frames> testingFrames)
{
    predict(testingFrames);


}

void testing::predict(std::vector<frames> predictionFrames)
{
    

    std::vector<matrix<rgb_pixel>> imagesTest;
    std::vector<unsigned long> labelsTest;
    dlib::rand rnd(time(0));
    int idx = 0;
    while (imagesTest.size() < 16)
    {
        
        //auto idx = rnd.get_random_32bit_number() % predictionFrames.size();
        //Loads frame into rgb_pixel matrix, resizes it to 224x224 and then adds to images vector
        matrix<rgb_pixel> obj;
        matrix<rgb_pixel> sizeimg(224, 224);
        dlib::load_image(obj, predictionFrames[idx].frameName);
        dlib::resize_image(obj, sizeimg);
        imagesTest.push_back(sizeimg);

        labelsTest.push_back(predictionFrames[idx].controls);
        idx++;
    }

    //Network load

    resnet::train_50 resnet50;
    //model::train net;
    dlib::deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50;
    auto backbone = std::move(resnet50.subnet().subnet());
    using net_type = loss_metric<fc<256, decltype(backbone)>>;
    net_type testing_nets;
  
    
    dlib::deserialize("game_classifier_network_resnet.dat") >> testing_nets;
    auto newback = std::move(testing_nets.subnet());
    using net_type2 = loss_multiclass_log< decltype(newback)>;
    net_type2 testing_net;
    std::vector<unsigned long> predicted_labels = testing_net(imagesTest);
    //softmax<net_type::subnet_type> anet;
    
    std::string binaryeqv;
    //Convert controls from binary to ValidInputs enum

    for (int i = 0; i < predicted_labels.size(); i++)
    {
        binaryeqv.clear();
        binaryeqv = std::bitset<8>(predicted_labels[i]).to_string();
        std::cout << predictionFrames[i].frameName << std::endl;
        printControls(binaryeqv);
       
        //emuHarness(binaryeqv, false);
        //emuHarness(binaryeqv, true);

    }
    //Print network state
    std::cout << testing_net << std::endl;
}


//Send keyboard inputs to windows host
/*
void testing::emuHarness(std::string input, bool releaseMode)
{
    
	 for(int i = 0; 0 < sizeof(input) / sizeof(input[0]); i++)
     {
         if(input[i] - '0' == 1)
            switch(i)
            {
                case 0:
                    ip.ki.wVk = VK_UP;
                    break;
                case 1:
                    ip.ki.wVk = VK_DOWN;
                    break;
                case 2:
                    ip.ki.wVk = VK_LEFT;
                    break;
                case 3:
                    ip.ki.wVk = VK_RIGHT;
                    break;
                case 4:
                    ip.ki.wVk = VK_LEFT;
                    break;
                case 5:
                    ip.ki.wVk = VK_LEFT;
                    break;
                case 6:

                case 7:

            }
            if (releaseMode)
            {
                ip.ki.dwFlags = KEYEVENTF_KEYUP;

            }
            else
            {
                ip.ki.dwFlags = 0;
            }
            SendInput(1, &ip, sizeof(INPUT));       
    }
}
*/

void testing::printControls(std::string binaryCtrls)
{
    std::vector<std::string> translatedCtrls;
    for (int i = 0; i < binaryCtrls.size(); i++)
    {
        if (binaryCtrls[i] - '0' == 1)
        {
            std::cout << ValidInputs[i] << std::endl;
            translatedCtrls.push_back(ValidInputs[i]);

        }

    }

}

/*
std::vector<rgb_pixel> testing::grabScreen()
{


}
*/



