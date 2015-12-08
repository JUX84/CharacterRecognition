cmake_minimum_required(VERSION 3.4)
project( CR )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O3 -DNDEBUG")
set(EXECUTABLE_OUTPUT_PATH bin)

add_executable( CR main.cpp )

target_link_libraries( CR ${OpenCV_LIBS} )
