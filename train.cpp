#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <vector>
#include <dlib/image_transforms.h>
#include "train.h"
#include <filesystem>
#include <Windows.h>

#pragma region Variables
using namespace dlib;

/*
// training network type
using net_type = loss_metric<fc_no_bias<16, avg_pool_everything<
    input_rgb_image
    >>>;

*/
/*
// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
    train::alevel0<
    train::alevel1<
    train::alevel2<
    train::alevel3<
    train::alevel4<
    max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
    input_rgb_image
    >>>>>>>>>>>>;
*/



//OBSOLETE
//TODO: Have a relative path
//std::string dir = "level1/1/";
//std::string dir2 = "level1/2/";

using net_type = loss_multiclass_log<fc<16,avg_pool_everything<
                            max_pool<3,3,2,2,relu<bn_con<con<64,7,7,2,2,
                            input_rgb_image>
                            >>>>>>;
template <typename SUBNET>

{
resnet::train_50 resnet50;
deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50;
}

using anet_type =
                  softmax<fc<5,
                  relu<fc<512,
                  input_rgb_image
                  >>>>;      


#pragma endregion


train::train(std::vector<frames> sortedFrames, std::vector<frames> valdiationFrames)
{
    trainer(sortedFrames, valdiationFrames);

}

int train::trainer(std::vector<frames> sortedFrames, std::vector<frames> validationFrames)
{

#pragma region net_definitions

    // Now, let's define the classic ResNet50 network and load the pretrained model on
    // ImageNet.
    resnet::train_50 resnet50;
    //std::vector<std::string> labels;
    deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50;
    auto backbone = resnet50.subnet();
    using anet_type<backbone> testnet;
    // For transfer learning, we are only interested in the ResNet50's backbone, which
    // lays below the loss and the fc layers, so we can extract it as:


    //auto backbone = resnet50.subnet();
    using net_type = loss_metric<fc_no_bias<16, decltype(backbone)>>;
    net_type net;
    softmax<net_type::subnet_type> net;
    auto anet = backbone;


    

    auto objs = sortedFrames;
    //dnn_trainer<net_type> trainer(net, sgd(0.0001, 0.9));
    dnn_trainer<net_type,adam> trainer(net, adam(0.0005, 0.9, 0.999));
    trainer.set_synchronization_file("resnet_game_sync", std::chrono::minutes(2));
    trainer.set_learning_rate(0.1);
    trainer.set_iterations_without_progress_threshold(1000);
    trainer.be_verbose();

#pragma endregion
    std::cout << "frame.size(): " << sortedFrames.size() << std::endl;
    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;

    regular_load(sortedFrames, images, labels);

    //Data batching code
    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<unsigned long> mini_batch_labels;
    dlib::rand rnd(time(0));
    // Loop until the trainer's automatic shrinking has shrunk the learning rate to 1e-6.
    // Given our settings, this means it will stop training after it has shrunk the
    // learning rate 3 times.
    std::cout << "Starting training" << std::endl;
    while (trainer.get_learning_rate() >= 1e-1)
    {
        mini_batch_samples.clear();
        mini_batch_labels.clear();

        // make a 128 image mini-batch
        while (mini_batch_samples.size() < 128)
        {
            auto idx = rnd.get_random_32bit_number() % images.size();
            mini_batch_samples.push_back(images[idx]);
            mini_batch_labels.push_back(labels[idx]);
        }

        // Tell the trainer to update the network given this mini-batch
        trainer.train_one_step(mini_batch_samples, mini_batch_labels);

        // You can also feed validation data into the trainer by periodically
        // calling trainer.test_one_step(samples,labels).  Unlike train_one_step(),
        // test_one_step() doesn't modify the network, it only computes the testing
        // error which it records internally.  This testing error will then be print
        // in the verbose logging and will also determine when the trainer's
        // automatic learning rate shrinking happens.  Therefore, test_one_step()
        // can be used to perform automatic early stopping based on held out data.   
    }
    
    trainer.get_net();
    std::cout << "Training complete" << std::endl;

    //Save weights
    net.clean();
    dlib::serialize("game_classifier_network_resnet.dat") << net;

    
#pragma region Testing_Validation

    
    std::vector<matrix<rgb_pixel>> imagesTest;
    std::vector<unsigned long> labelsTest;
    //dlib::rand rnd(time(0));

    while (imagesTest.size() < 64)
    {
        auto idx = rnd.get_random_32bit_number() % validationFrames.size();
        //Loads frame into rgb_pixel matrix, resizes it to 224x224 and then adds to images vector
        matrix<rgb_pixel> obj;
        matrix<rgb_pixel> sizeimg(224, 224);
        dlib::load_image(obj, validationFrames[idx].frameName);
        dlib::resize_image(obj, sizeimg);
        imagesTest.push_back(sizeimg);

        labelsTest.push_back(validationFrames[idx].controls);
    }


    // Now, just to show an example of how you would use the network, let's check how well
    // it performs on the training data.

    // Normally you would use the non-batch-normalized version of the network to do
    // testing, which is what we do here.
    auto testing_net = net;

    // Run all the images through the network to get their vector embeddings.
    std::vector<matrix<float, 0, 1>> embedded = testing_net(imagesTest);

    // Now, check if the embedding puts images with the same labels near each other and
    // images with different labels far apart.
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < embedded.size(); ++i)
    {
        for (size_t j = i + 1; j < embedded.size(); ++j)
        {
            if (labelsTest[i] == labelsTest[j])
            {
                // The loss_metric layer will cause images with the same label to be less
                // than net.loss_details().get_distance_threshold() distance from each
                // other.  So we can use that distance value as our testing threshold.
                if (length(embedded[i] - embedded[j]) < testing_net.loss_details().get_distance_threshold())
                    ++num_right;
                else
                    ++num_wrong;
            }
            else
            {
                if (length(embedded[i] - embedded[j]) >= testing_net.loss_details().get_distance_threshold())
                    ++num_right;
                else
                    ++num_wrong;
            }
        }
    }

    std::cout << "num_right: " << num_right << std::endl;
    std::cout << "num_wrong: " << num_wrong << std::endl;

#pragma endregion













    return 0;
}

void train::regular_load(std::vector<frames>& sortedFrames, std::vector<matrix<rgb_pixel>>& images, std::vector<unsigned long>& labels)
{
    for (int i = 0; i < sortedFrames.size(); i++)
    {
        //Loads frame into rgb_pixel matrix, resizes it to 224x224 and then adds to images vector
        matrix<rgb_pixel> obj;
        matrix<rgb_pixel> sizeimg(224, 224);
        dlib::load_image(obj, sortedFrames[i].frameName);
        dlib::resize_image(obj, sizeimg);
        images.push_back(sizeimg);
        labels.push_back(sortedFrames[i].controls);
    }
}










