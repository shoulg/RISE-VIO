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

#ifndef GVINS_FEATURE_H
#define GVINS_FEATURE_H

#include <memory>
#include <opencv2/opencv.hpp>

using cv::Mat;

class Frame;
class MapPoint;

enum FeatureType {
    FEATURE_NONE             = -1,
    FEATURE_MATCHED          = 0,
    FEATURE_TRIANGULATED     = 1,
    FEATURE_DEPTH_ASSOCIATED = 2,
};

class Feature {

public:
    typedef std::shared_ptr<Feature> Ptr;

    Feature() = delete;
//here
    Feature(const std::shared_ptr<Frame> &frame, const Eigen::Vector2d &velocity, Vector3d point,
            cv::Point2f keypoint, cv::Point2f distorted, FeatureType type, double time)
        : frame_(frame)
        , point_(point)
        , keypoint_(std::move(keypoint))
        , distorted_keypoint_(std::move(distorted))
        , isoutlier_(false)
        , type_(type) 
        , feature_time_(time){

        velocity_[0] = velocity[0];
        velocity_[1] = velocity[1];
        velocity_[2] = 0;

    }

    static std::shared_ptr<Feature> createFeature(const std::shared_ptr<Frame> &frame, const Eigen::Vector2d &velocity,
                                                  const Vector3d &point, const cv::Point2f &keypoint, const cv::Point2f &distorted,
                                                  FeatureType type, double time) {
        return std::make_shared<Feature>(frame, velocity, point, keypoint, distorted, type, time);
    }

    std::shared_ptr<Frame> getFrame() {
        return frame_.lock();
    }

    std::shared_ptr<MapPoint> getMapPoint() {
        return mappoint_.lock();
    }

    const cv::Point2f &keyPoint() {
        return keypoint_;
    }

    const cv::Point2f &distortedKeyPoint() {
        return distorted_keypoint_;
    }

    void addMapPoint(const std::shared_ptr<MapPoint> &mappoint) {
        mappoint_ = mappoint;
    }

    void setOutlier(bool isoutlier) {
        isoutlier_ = isoutlier;
    }

    bool isOutlier() const {
        return isoutlier_;
    }

    FeatureType featureType() {
        return type_;
    }

    const Vector3d &velocityInPixel() {
        return velocity_;
    }

    void setVelocityInPixel(const cv::Point2f &velocity) {
        velocity_[0] = velocity.x;
        velocity_[1] = velocity.y;
        velocity_[2] = 0;
    }

    //drt_vio_init add
    Eigen::Vector3d point() const {
        return point_;
    }

    cv::Point2f point_2d() const {
        cv::Point2f point_2d;
        point_2d.x = point_[0];
        point_2d.y = point_[1];
        return point_2d;
    }

    Eigen::Vector2d uv() const {
        Eigen::Vector2d uv;
        uv[0] = keypoint_.x;
        uv[1] = keypoint_.y;
        return uv;
    }

private:
    std::weak_ptr<Frame> frame_;
    std::weak_ptr<MapPoint> mappoint_;

    Vector3d point_;
    cv::Point2f keypoint_, distorted_keypoint_;
    Vector3d velocity_{0, 0, 0};

    bool isoutlier_;

    FeatureType type_;

    //drt_vio_init add
    double feature_time_;
};

#endif // GVINS_FEATURE_H
