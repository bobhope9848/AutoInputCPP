#include "testing.h"
#include <vector>
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <Windows.h>
#include <array>

testing::testing()
{
	


}

testing::testing(std::vector<frames> testingFrames)
{



}

std::vector<testing::ValidInputs> testing::predict(std::string modelLoc)
{
    //Network load

    resnet::train_50 resnet50;
    //std::vector<std::string> labels;
    deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50;
    // For transfer learning, we are only interested in the ResNet50's backbone, which
    // lays below the loss and the fc layers, so we can extract it as:
    auto backbone = resnet50.subnet().subnet();
    using net_type = loss_metric<fc_no_bias<16, decltype(backbone)>>;
    net_type net;
    deserialize(modelLoc + "game_classifier_network_resnet.dat") >> net;
    
    softmax<net_type::subnet_type> anet;
    
    auto anet = backbone;






	return std::vector<ValidInputs>();
}

void testing::emuHarness(std::vector<ValidInputs> input)
{
	 for(int i = 0; 0 < sizeof(input) / sizeof(input[0]); i++)
    {
        switch(input[i])
        {
            case left:
                inputs[i].type = INPUT_KEYBOARD;
                inputs[i].ki.wVk = VK_LEFT;
                break;
            case right:
                inputs[i].type = INPUT_KEYBOARD;
                inputs[i].ki.wVk = VK_RIGHT;
                break;
            case up:
                inputs[i].type = INPUT_KEYBOARD;
                inputs[i].ki.wVk = VK_UP;
                break;
            case down:
                inputs[i].type = INPUT_KEYBOARD;
                inputs[i].ki.wVk = VK_DOWN;
                break;
            case start:
                inputs[i].type = INPUT_KEYBOARD;
                inputs[i].ki.wVk = VK_LEFT;
                break;
            case select:
                inputs[i].type = INPUT_KEYBOARD;
                inputs[i].ki.wVk = VK_LEFT;
                break;
            case a:

            case b:
            case begin:
            case end:

        }
    }
    UINT uSent = SendInput(ARRAYSIZE(inputs), inputs, sizeof(INPUT));
}

void testing::releaseCtrls()
{

}




