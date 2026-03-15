cmake_minimum_required(VERSION 3.10)
project(rb_vins)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

# Optimization
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# C/C++ 语言编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native -funroll-loops -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -funroll-loops -flto -DNDEBUG")

# Build GVINS library
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/rb_vins)
file(GLOB_RECURSE GVINS_SOURCE
        rb_vins/*.cc
        rb_vins/common/*.cc
        rb_vins/fileio/*.cc
        rb_vins/tracking/*.cc
        rb_vins/preintegration/*.cc
        rb_vins/morefuncs/*.cc)
add_library(rb_vins_core SHARED
        ${GVINS_SOURCE})

# OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(rb_vins_core ${OpenCV_LIBS})

# glog
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Glog REQUIRED)
target_link_libraries(rb_vins_core ${GLOG_LIBRARIES})

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
target_link_libraries(rb_vins_core ${YAML_CPP_LIBRARIES})

# Ceres
find_package(Ceres REQUIRED)
target_link_libraries(rb_vins_core ${CERES_LIBRARIES})

# boost
find_package(Boost REQUIRED COMPONENTS filesystem)
target_link_libraries(rb_vins_core
        ${Boost_LIBRARIES})

# TBB
find_package(TBB REQUIRED)
target_link_libraries(rb_vins_core
        TBB::tbb)

# abseil
set(ABSL_PROPAGATE_CXX_STD true)
add_subdirectory(thirdparty/abseil-cpp abseil-cpp)
target_link_libraries(rb_vins_core
        absl::strings
        absl::str_format
        absl::time)

# # 查找 iir 库 # filter add
# find_library(IIR_LIBRARY iir HINTS /usr/local/lib)
# if(NOT IIR_LIBRARY)
#     message(FATAL_ERROR "Could not find iir library")
# endif()

# ROS interface
find_package(catkin REQUIRED COMPONENTS
        message_generation
        roscpp
        std_msgs)
catkin_package(
        LIBRARIES
        CATKIN_DEPENDS roscpp std_msgs)
include_directories(
        ${catkin_INCLUDE_DIRS})

# GVINS ROS with standard messages
add_executable(${PROJECT_NAME}_ros
        ROS/fusion_ros.cc
        ROS/drawer_rviz.cc)

        add_dependencies(${PROJECT_NAME}_ros ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_ros
        ${catkin_LIBRARIES}
        ${IIR_LIBRARY}
        rb_vins_core)
