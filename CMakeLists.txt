cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(predict)

#set(CMAKE_TOOLCHAIN_FILE "C:/Users/yusuke/Documents/College Work/CSCI 24000/Try again/cmake please/vcpkg/scripts/buildsystems" )
set(dlib_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dlibDebug/lib/cmake/dlib/" PATH)


#find_package(OpenCV REQUIRED COMPONENTS opencv_highgui)
#find_package(dlib REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dlibDebug/lib/cmake/dlib/)
#find_package(dlib REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dlibRelease/lib/cmake/dlib/)
find_package(dlib REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dlibDebugCUDA/lib/cmake/dlib/)
#find_package(dlib REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dlibReleaseCUDA/lib/cmake/dlib/)

add_executable(predict 
	"dispatch.cpp" 
	include/resnet.h
	"frames.h"
	"train.h"
	"train.cpp"
	"testing.cpp"
	"testing.h"
	"model.cpp"
	"model.h"
)
	# "testing.cpp" "model.cpp" "model.h" "testing.h")

#target_link_libraries(predict PUBLIC dlib::dlib PRIVATE opencv_highgui)
target_link_libraries(predict dlib::dlib)

add_custom_command(
        TARGET predict POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/resnet50_1000_imagenet_classifier.dnn
                ${CMAKE_CURRENT_BINARY_DIR}/resnet50_1000_imagenet_classifier.dnn)

