cmake_minimum_required(VERSION 3.4)
project( CR )

IF(MSVC)
	set(OpenCV_INCLUDE_DIRS "C:/opencv/build/x86/vc12")
	set(OpenCV_LIBS_DIRS "C:/opencv/build/x86/vc12/lib")
	set(OpenCV_LIBS "${OpenCV_LIBS_DIRS}/opencv_ts300.lib ${OpenCV_LIBS_DIRS}/opencv_ts300.lib")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -DNDEBUG")
ELSE()
	find_package( OpenCV REQUIRED )
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -O3 -DNDEBUG")
ENDIF()

include_directories( inc ${OpenCV_INCLUDE_DIRS} )

set(EXECUTABLE_OUTPUT_PATH bin)

add_executable(CR src/main.cpp src/characterRecognizer.cpp src/logger.cpp)

target_link_libraries(CR ${OpenCV_LIBS})
