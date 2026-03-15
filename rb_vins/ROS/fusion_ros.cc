/*
 * IC-GVINS: A Robust, Real-time, INS-Centric GNSS-Visual-Inertial Navigation System
 *
 * Copyright (C) 2022 i2Nav Group, Wuhan University
 *
 *     Author : Hailiang Tang
 *    Contact : thl@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "fusion_ros.h"
#include "drawer_rviz.h"

#include "rb_vins/common/angle.h"
#include "rb_vins/common/logging.h"
#include "rb_vins/misc.h"
#include "rb_vins/tracking/frame.h"

#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <sensor_msgs/image_encodings.h>

#include <atomic>
#include <csignal>
#include <memory>

std::atomic<bool> isfinished{false};

void sigintHandler(int sig);
void checkStateThread(std::shared_ptr<FusionROS> fusion);

void FusionROS::setFinished() {
    if (gvins_ && gvins_->isRunning()) {
        gvins_->setFinished();
    }
}

void FusionROS::run() {
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // message topic
    string imu_topic, image_topic;
    pnh.param<string>("imu_topic", imu_topic, "/imu0");
    pnh.param<string>("image_topic", image_topic, "/cam0");

    // GVINS parameter
    string configfile;
    pnh.param<string>("configfile", configfile, "gvins.yaml");

    // Load configurations
    YAML::Node config;
    std::vector<double> vecdata;
    try {
        config = YAML::LoadFile(configfile);
    } catch (YAML::Exception &exception) {
        std::cout << "Failed to open configuration file" << std::endl;
        return;
    }
    auto outputpath        = config["outputpath"].as<string>();
    auto is_make_outputdir = config["is_make_outputdir"].as<bool>();

    // Create the output directory
    if (!boost::filesystem::is_directory(outputpath)) {
        boost::filesystem::create_directory(outputpath);
    }
    if (!boost::filesystem::is_directory(outputpath)) {
        std::cout << "Failed to open outputpath" << std::endl;
        return;
    }

    if (is_make_outputdir) {
        absl::CivilSecond cs = absl::ToCivilSecond(absl::Now(), absl::LocalTimeZone());
        absl::StrAppendFormat(&outputpath, "/T%04d%02d%02d%02d%02d%02d", cs.year(), cs.month(), cs.day(), cs.hour(),
                              cs.minute(), cs.second());
        boost::filesystem::create_directory(outputpath);
    }

    // Glog output path
    FLAGS_log_dir = outputpath;

    // The GVINS object
    Drawer::Ptr drawer = std::make_shared<DrawerRviz>(nh);
    gvins_             = std::make_shared<GVINS>(configfile, outputpath, drawer);

    // check is initialized
    if (!gvins_->isRunning()) {
        LOGE << "Fusion ROS terminate";
        return;
    }

    // subscribe message
    ros::Subscriber imu_sub   = nh.subscribe<sensor_msgs::Imu>(imu_topic, 1000, &FusionROS::imuCallback, this, ros::TransportHints().tcpNoDelay(true));
    ros::Subscriber image_sub = nh.subscribe<sensor_msgs::Image>(image_topic, 50, &FusionROS::imageCallback, this, ros::TransportHints().tcpNoDelay(true));

    LOGI << "Waiting ROS message...";

    // enter message loopback
    ros::spin();
}

void FusionROS::imuCallback(const sensor_msgs::ImuConstPtr &imumsg) {
    // Time convertion
    double unixsecond = imumsg->header.stamp.toSec();

    if (!imu_initialized) {
        imu_.time = unixsecond;
        imu_pre_ = imu_;
        imu_initialized = true;
        return;
    }

    imu_.time = unixsecond;
    // delta time
    imu_.dt = imu_.time - imu_pre_.time;

    // IMU measurements, Front-Right-Down
    imu_.dtheta[0] = imumsg->angular_velocity.x * imu_.dt;
    imu_.dtheta[1] = imumsg->angular_velocity.y * imu_.dt;
    imu_.dtheta[2] = imumsg->angular_velocity.z * imu_.dt;
    imu_.dvel[0]   = imumsg->linear_acceleration.x * imu_.dt;
    imu_.dvel[1]   = imumsg->linear_acceleration.y * imu_.dt;
    imu_.dvel[2]   = imumsg->linear_acceleration.z * imu_.dt;

    imu_pre_ = imu_;

    imu_buffer_.push(imu_);
    while (!imu_buffer_.empty()) {
        auto imu = imu_buffer_.front();

        // Add new IMU to GVINS
        if (gvins_->addNewImu(imu)) {
            imu_buffer_.pop();
        } else {
            // Thread lock failed, try next time
            break;
        }
    }
}

void FusionROS::imageCallback(const sensor_msgs::ImageConstPtr &imagemsg) {
    // Mat image;

    // // Copy image data
    // if (imagemsg->encoding == sensor_msgs::image_encodings::MONO8) {
    //     image = Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC1);
    //     memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width);
    // } else if (imagemsg->encoding == sensor_msgs::image_encodings::BGR8) {
    //     image = Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC3);
    //     memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width * 3);
    // }
    cv::Mat image;
    if (imagemsg->encoding == sensor_msgs::image_encodings::MONO8) {
        image = Mat(imagemsg->height, imagemsg->width, CV_8UC1);
        if (imagemsg->data.size() >= imagemsg->height * imagemsg->width) {
            memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width);
        } else {
            ROS_ERROR("Image data size too small");
            return;
        }
    } else if (imagemsg->encoding == sensor_msgs::image_encodings::BGR8 ||
                   imagemsg->encoding == sensor_msgs::image_encodings::RGB8) {
        image = Mat(imagemsg->height, imagemsg->width, CV_8UC3);
        if (imagemsg->data.size() >= imagemsg->height * imagemsg->width * 3) {
            memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width * 3);

            // 转灰度
            if (imagemsg->encoding == sensor_msgs::image_encodings::RGB8)
                cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
            else
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        } else {
            ROS_ERROR("Image data size too small");
            return;
        }
    } else {
        ROS_ERROR("Unsupported image encoding: %s", imagemsg->encoding.c_str());
        return;
    }

    // Time convertion
    double unixsecond = imagemsg->header.stamp.toSec();

    // Add new Image to GVINS
    frame_ = Frame::createFrame(unixsecond, image);

    frame_buffer_.push(frame_);
    while (!frame_buffer_.empty()) {
        auto frame = frame_buffer_.front();
        if (gvins_->addNewFrame(frame)) {
            frame_buffer_.pop();
        } else {
            break;
        }
    }

    LOG_EVERY_N(INFO, 20) << "Raw data time " << Logging::doubleData(imu_.time) << ", "
                          << Logging::doubleData(frame_->stamp());
}

void sigintHandler(int sig) {
    std::cout << "Terminate by Ctrl+C " << sig << std::endl;
    isfinished = true;
}

void checkStateThread(std::shared_ptr<FusionROS> fusion) {
    std::cout << "Check thread is started..." << std::endl;

    auto fusion_ptr = std::move(fusion);
    while (!isfinished) {
        sleep(1);
    }

    // Exit the GVINS thread
    fusion_ptr->setFinished();

    std::cout << "GVINS has been shutdown ..." << std::endl;

    // Shutdown ROS
    ros::shutdown();

    std::cout << "ROS node has been shutdown ..." << std::endl;
}

int main(int argc, char *argv[]) {
    // Glog initialization
    Logging::initialization(argv, true, true);

    // ROS node
    ros::init(argc, argv, "gvins_node", ros::init_options::NoSigintHandler);

    // Register signal handler
    std::signal(SIGINT, sigintHandler);

    auto fusion = std::make_shared<FusionROS>();

    // Check thread
    std::thread check_thread(checkStateThread, fusion);

    std::cout << "Fusion process is started..." << std::endl;

    // Enter message loop
    fusion->run();

    return 0;
}
