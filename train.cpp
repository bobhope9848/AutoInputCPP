#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <vector>
#include <dlib/image_transforms.h>
#include "train.h"
#include <filesystem>
#include <Windows.h>

#pragma region Variables





bool justTest = false;



#pragma endregion


train::train(std::vector<frames> sortedFrames)
{
    trainer(sortedFrames);

}

int train::trainer(std::vector<frames> sortedFrames)
{

#pragma region net_definitions

    if (!justTest)
    {
        //set_dnn_prefer_smallest_algorithms();


        resnet::train_50 resnet50;
        deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50;
        //Extract resnet backbone
        auto backbone = std::move(resnet50.subnet().subnet());
       
        //Slap on fully connected layer and loss metric layer on top of backbone.
        using net_type = loss_metric<fc<256, decltype(backbone)>>;
        net_type net;

        //dnn_trainer<net_type> trainer(net, sgd(0.0001, 0.9));
        dnn_trainer<net_type, adam> trainer(net, adam(0.0005, 0.9, 0.999));
        trainer.set_synchronization_file("resnet_game_sync", std::chrono::minutes(4));
        trainer.set_learning_rate(0.1);
        trainer.set_iterations_without_progress_threshold(1000);
        trainer.be_verbose();

#pragma endregion
        std::cout << "frame.size(): " << sortedFrames.size() << std::endl;
        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        time_t seed = 0;
        int batchsize = 16;
        //regular_load(sortedFrames, images, labels);

        //Data batching code
        std::vector<matrix<rgb_pixel>> mini_batch_samples;
        std::vector<unsigned long> mini_batch_labels;
        dlib::rand rnd(time(0));
        
        std::cout << "Starting training" << std::endl;
        //Load in frames into memory (CAREFUL, LARGE SIZE)
        regularLoad(sortedFrames.size(), rnd, sortedFrames, images, labels);
        
        //Run until learning ratehas shrunk from 0.1 to 0.00001
        while (trainer.get_learning_rate() >= 1e-4)
        {
            mini_batch_samples.clear();
            mini_batch_labels.clear();
            dlib::rand rnd(time(0) + seed);

            while (mini_batch_samples.size() < 16)
            {
                auto idx = rnd.get_random_32bit_number() % images.size();
                mini_batch_samples.push_back(images[idx]);
                mini_batch_labels.push_back(labels[idx]);
            }


            // Tell the trainer to update the network given this mini-batch
            trainer.train_one_step(mini_batch_samples, mini_batch_labels);

        }

        trainer.get_net();
        std::cout << "Training complete" << std::endl;

        //Save weights
        net.clean();
        dlib::serialize("game_classifier_network_resnet.dat") << net;
    }
    
#pragma endregion
    

    return 0;
}

void train::regularLoad(int batchSize, dlib::rand& rnd, std::vector<frames>& sortedFrames, std::vector<matrix<rgb_pixel>>& images, std::vector<unsigned long>& labels)
{
    //int i;
    for (int i = 0; images.size() < batchSize; i++)
    {
        //size_t id = rnd.get_random_32bit_number() % sortedFrames.size();
        //i = rnd.get_random_32bit_number() % sortedFrames.size();
        //Loads frame into rgb_pixel matrix, resizes it to 224x224 and then adds to images vector
        matrix<rgb_pixel> obj;
        matrix<rgb_pixel> sizeimg(224, 224);
        dlib::load_image(obj, sortedFrames[i].frameName);
        //assign_all_pixels(obj, rgb_pixel(0, 0, 0));
        dlib::resize_image(obj, sizeimg);
        images.push_back(sizeimg);
        labels.push_back(sortedFrames[i].controls);
    }

}










