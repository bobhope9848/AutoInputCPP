#ifndef TRAIN_H
#define TRAIN_H


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include "include/resnet.h"
#include "frames.h"

using namespace dlib;
class train
{

private:
protected:

public:
	
	// The next page of code defines a ResNet network.  It's basically copied
	// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
	// layer with loss_metric and make the network somewhat smaller.
	
	template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
	using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;
	
	template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
	using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;
	
	template <int N, template <typename> class BN, int stride, typename SUBNET>
	using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
	
	
	template <int N, typename SUBNET> using res = relu<residual<block, N, bn_con, SUBNET>>;
	template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
	template <int N, typename SUBNET> using res_down = relu<residual_down<block, N, bn_con, SUBNET>>;
	template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;
	
	// ----------------------------------------------------------------------------------------
	
	template <typename SUBNET> using level0 = res_down<32, SUBNET>;
	template <typename SUBNET> using level1 = res<256, res<256, res_down<256, SUBNET>>>;
	template <typename SUBNET> using level2 = res<128, res<128, res_down<128, SUBNET>>>;
	template <typename SUBNET> using level3 = res<64, res<64, res<64, res_down<64, SUBNET>>>>;
	template <typename SUBNET> using level4 = res<32, res<32, res<32, SUBNET>>>;
	
	template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
	template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
	template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
	template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
	template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
	
	
	train(std::vector<frames> sortedFrames, std::vector<frames> validationFrames);

	int trainer(std::vector<frames> sortedFrames, std::vector<frames> validationFrames);
	void regular_load(std::vector<frames>& sortedFrames, std::vector<matrix<rgb_pixel>>& images, std::vector<unsigned long>& labels);

	

};

#endif