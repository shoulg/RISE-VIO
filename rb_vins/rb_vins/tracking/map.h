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

#ifndef GVINS_MAP_H
#define GVINS_MAP_H

#include "tracking/camera.h"
#include "tracking/frame.h"
#include "tracking/mappoint.h"
#include "preintegration/integration_state.h"
#include "preintegration/preintegration.h"

#include "../common/eigentypes.h"

#include <memory>
#include <mutex>
#include <unordered_map>

//drt_vio_init add
class FeaturePerFrame
{
  public:
    FeaturePerFrame() = default;
    FeaturePerFrame(double time, const Vector3d &point, const cv::Point2f &pts2d_undis, 
                    const cv::Point2f &pts2d, const Vector2d &_velocity)
    : time_frame_id_(time),
      point_(std::move(point)),
      uv_undis_(std::move(pts2d_undis)),
      uv_(pts2d),
      velocity_(std::move(_velocity)),
      z_(-1.0),
      parallax_(0.0)
    {}

    Vector3d point() const {
        return point_;
    }

    Vector2d keyPoint() const {
        Vector2d keypoint;
        keypoint[0] = uv_undis_.x;
        keypoint[1] = uv_undis_.y;
        return keypoint;
    }

    Vector2d velocity() const {
        return velocity_;
    }

    Vector3d velocity3d() const {
        Vector3d vel;
        vel[0] = velocity_[0];
        vel[1] = velocity_[1];
        vel[2] = 1;
        return vel;
    }

    void setvelocity(Vector2d velocity) {
        velocity_ = std::move(velocity);
    }

    cv::Point2f uv_undis() const {
        return uv_undis_;
    }

    cv::Point2f uv() const {
        return uv_;
    }

    double time_frame_id_;
    Vector3d point_;
    cv::Point2f uv_undis_;
    cv::Point2f uv_;
    Vector2d velocity_;

    double z_;
    double parallax_;
};

class drtSFMFeature {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using id_type = uint32_t;  // 根据需求也可使用更小的数据类型

    drtSFMFeature() = default;

    //接受 FeaturePerFrame 的右值引用，利用移动语义
    drtSFMFeature(double _start_frame, FeaturePerFrame &&feature_per_frame)
    : id(global_id_counter++),
      start_frame_(_start_frame),
      obs{{_start_frame, std::move(feature_per_frame)}},
      estimated_depth(-1.0)
    {}

    // 唯一 id
    id_type id;
    double start_frame_{};

    bool state = true; // 是否加入LTL
    Vector3d p3d; // sfm使用：世界坐标系下的3D位置
    // TODO: 观测值
    Eigen::aligned_map<double, FeaturePerFrame> obs;
    double estimated_depth;

    // 声明静态变量
    static std::atomic<id_type> global_id_counter;
};

class Map {

public:
    typedef std::shared_ptr<Map> Ptr;

    typedef std::unordered_map<ulong, Frame::Ptr> KeyFrames;
    typedef std::unordered_map<ulong, MapPoint::Ptr> LandMarks;

    Map() = delete;
    explicit Map(size_t size, int initial_triangulate_parallax)
        : window_size_(size), initial_triangulate_parallax_(initial_triangulate_parallax) {
    }

    void resetWindowSize(size_t size) {
        window_size_ = size;
    }

    size_t windowSize() const {
        return window_size_;
    }

    void insertKeyFrame(const Frame::Ptr &frame);

    const KeyFrames &keyframes() {
        return keyframes_;
    }

    // tmp change
    const KeyFrames &subkeyframes() {
        return nonkeyframes_;
    }

    const LandMarks &landmarks() {
        return landmarks_;
    }

    vector<ulong> orderedKeyFrames();
    Frame::Ptr oldestKeyFrame();
    const Frame::Ptr &latestKeyFrame();

    void removeMappoint(MapPoint::Ptr &mappoint);
    void removeKeyFrame(Frame::Ptr &frame, bool isremovemappoint);

    double mappointObservedRate(const MapPoint::Ptr &mappoint);

    bool isMaximumKeframes() {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return keyframes_.size() > window_size_;
    }

    bool isKeyFrameInMap(const Frame::Ptr &frame) {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return keyframes_.find(frame->keyFrameId()) != keyframes_.end();
    }

    // tmp change
    bool isValidKeyFrame(const Frame::Ptr &frame) {
        std::unique_lock<std::mutex> lock(map_mutex_);
        double oldest_time = timeToKeyframeId_.begin()->first;
        return (frame->stamp() > (oldest_time - 0.001));
    }
    bool isSubKeyFrameInMap(const Frame::Ptr &frame) {
        return nonkeyframes_.find(frame->keyFrameId()) != nonkeyframes_.end();
    }

    bool isWindowFull() {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return is_window_full_;
    }

    bool isWindowNormal() {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return keyframes_.size() == window_size_;
    }

    //drt_vio_init add
    void setCamera(std::shared_ptr<Camera> camera) {
        camera_ = camera;  // 直接赋值，不需要 std::move()
    }

    void updateSFMConstruct(double cur_time, std::vector<drtSFMFeature>& construct);

    std::pair<std::unique_lock<std::mutex>, const Eigen::aligned_unordered_map<uint32_t, drtSFMFeature>&> SFMConstruct() {
        std::unique_lock<std::mutex> lock(sfm_mutex_);
        return {std::move(lock), SFMConstruct_};  // 返回锁+引用
    }

    std::pair<std::unique_lock<std::mutex>, Eigen::aligned_unordered_map<uint32_t, drtSFMFeature>&> SFMConstructMutable() {
        std::unique_lock<std::mutex> lock(sfm_mutex_);
        return {std::move(lock), SFMConstruct_};  // 返回锁+可写引用
    }

    void retriangulate(const std::deque<std::pair<double, Pose>>& poselist);
    Eigen::Matrix4d pose2Tcw(const Pose &pose);
    void triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                          const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw);
    bool isGoodToTrack(const cv::Point2f &pp, const Pose &pose, const Vector3d &pw, double scale, double depth_scale);
    bool isGoodDepth(double depth, double scale);
    double keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0, const Pose &pose1);

    void moveKeyFrameToNonKeyFrame(const Frame::Ptr &frame);
    void moveNewestKeyFrameToNonKeyFrame(const Frame::Ptr &frame);
    void clearSubMap();
    void retriangulate_old();
    void removesfmfeaturebyid(std::vector<uint32_t> &removesfm_ids);
    Frame::Ptr getframebytime(double time);

private:
    Frame::Ptr getFrameByTimeNoLock(double time);

    std::mutex map_mutex_;

    KeyFrames keyframes_;
    //tmp change
    KeyFrames nonkeyframes_;
    
    LandMarks landmarks_;

    Frame::Ptr latest_keyframe_;

    size_t window_size_{20};
    bool is_window_full_{false};
    int initial_triangulate_parallax_ = 15;

    //drt_vio_init add
    std::shared_ptr<Camera> camera_;
    std::mutex sfm_mutex_;
    std::mutex init_mutex_;
    bool is_init_{false};
    Eigen::aligned_unordered_map<uint32_t, drtSFMFeature> SFMConstruct_;
    std::map<double, ulong> timeToKeyframeId_;
};

#endif // GVINS_MAP_H
