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

#ifndef GVINS_TRACKING_H
#define GVINS_TRACKING_H

#include "tracking/camera.h"
#include "tracking/drawer.h"
#include "tracking/feature.h"
#include "tracking/frame.h"
#include "tracking/map.h"
#include "tracking/mappoint.h"

#include "common/timecost.h"
#include "fileio/filesaver.h"

#include "morefuncs/GNC_EPNP.h"

#include <fstream>
#include <optional>

typedef enum TrackState {
    TRACK_FIRST_FRAME,
    TRACK_INITIALIZING,
    TRACK_TRACKING,
    TRACK_PASSED,
    TRACK_CHECK,
    TRACK_LOST,
} TrackState;

class Tracking {

public:
    typedef std::shared_ptr<Tracking> Ptr;

    enum TrackChoose {
        TRACK_C_PREINITIALING,
        TRACK_C_INITIALING,
        TRACK_C_INITIALIZED,
    };

    Tracking(Camera::Ptr camera, Map::Ptr map, Drawer::Ptr drawer, const string &configfile, const string &outputpath);

    TrackState track(Frame::Ptr frame);

    bool isNewKeyFrame() const {
        return isnewkeyframe_;
    }

    bool isGoodToTrack(const cv::Point2f &pp, const Pose &pose, const Vector3d &pw, double scale,
                       double depth_scale = 1.0);
    static Eigen::Matrix4d pose2Tcw(const Pose &pose);

    void setTrackChooseInitialing() {
        if (trackchoose_ < TRACK_C_INITIALING) {
            trackchoose_ = TRACK_C_INITIALING;
        }
        ulong thr = frame_cur_->id() + offset_;
        scheduled_threshold_.store(thr, std::memory_order_release);
    }
    // 提取 Pose，移动出去
    std::optional<std::tuple<double, Pose, Eigen::Matrix<double, 6, 6>>> extractPoseprior() {
        auto tmp = std::move(pose_cur_prior_);
        pose_cur_prior_ = std::nullopt;  // 显式清空
        return tmp;
    }

    void gettracknum();

private:
    void showTracking();

    bool preprocessing(Frame::Ptr frame);
    static double calculateHistigram(const Mat &image);

    void makeNewFrame(int state);
    void writeLoggingMessage();

    bool doResetTracking();

    static bool isGoodDepth(double depth, double scale = 1.0);

    bool trackReferenceFrame();
    bool trackReferenceFrame_old();//add
    bool trackReferenceFrame_both();//add
    bool trackMappoint();
    bool trackMappoint_both();//add

    double relativeTranslation();
    double relativeRotation();

    keyFrameState checkKeyFrameSate();

    int parallaxFromReferenceKeyPoints(const vector<cv::Point2f> &ref, const vector<cv::Point2f> &cur,
                                       double &parallax);
    int parallaxFromReferenceMapPoints(double &parallax);

    double keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0, const Pose &pose1);

    void featuresDetection(Frame::Ptr &frame, bool ismask = true);

    bool triangulation();
    bool triangulation(const Pose &gncpose);
    static void triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                                 const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw);

    bool isOnBorder(const cv::Point2f &pts);
    inline double ptsDistance(const cv::Point2f &pt1, const cv::Point2f &pt2) {
        return std::hypot(pt1.x - pt2.x, pt1.y - pt2.y);
    }

    template <typename T> static void reduceVector(T &vec, const vector<uint8_t> &status);

    //drt_vio_init add
    void addConstructobs(const vector<cv::Point2f> &pts2d);
    void updateConstruct();
    void makeNewFrame_old(int state);
    bool movelastkeyframe();
    void checkmappoint();
    std::optional<Pose> checkgncpose();
    void predictNextPose();
    void remove2dpoints();
    void updateremoveid(const std::vector<unsigned char> &status);

public:
    // 三点平面拟合的最大深度差异
    static constexpr double ASSOCIATE_MAXIUM_DISTANCE = 1.0;

    static constexpr double ASSOCIATE_MAXIUM_DISTANCE_RATE = 0.05;

    static constexpr double ASSOCIATE_DEPTH_STD = 0.1;

private:
    // 配置参数
    // Configurations for visual process
    const double TRACK_BLOCK_SIZE   = 200; // 特征提取分块大小 200.0 240.0(viode)
    const int TRACK_PYRAMID_LEVEL   = 3;     // 光流跟踪金字塔层数
    const double TRACK_MIN_PARALLAX = 10.0;  // 三角化时最小的像素视差
    const double TRACK_MIN_INTERVAl = 0.08;  // 最短观测帧时间间隔0.08

    double feature_density_scale_ = 2.0;
    double initial_deltatime_ = 0.22;
    int triangulate_parallax_ = 15;

    ulong offset_ = 0;
    std::atomic<ulong> scheduled_threshold_; // 要达到的关键帧id阈值

    // 图像帧
    Frame::Ptr frame_cur_, frame_ref_, frame_pre_, last_keyframe_;
    // tmp change
    Frame::Ptr frame_ppre_;
    
    Camera::Ptr camera_;
    Map::Ptr map_;
    Drawer::Ptr drawer_;

    // 直方图均衡化
    cv::Ptr<cv::CLAHE> clahe_;

    // 特征点
    // For feature tracking
    vector<cv::Point2f> pts2d_cur_, pts2d_new_, pts2d_ref_;
    vector<Frame::Ptr> pts2d_ref_frame_;

    vector<Eigen::Vector2d> velocity_ref_, velocity_cur_;

    vector<MapPoint::Ptr> tracked_mappoint_, mappoint_matched_;

    // 分块特征提取, 第一个为分块的长宽, 然后是按照一行一行的分块起始坐标
    // For feature detection
    vector<std::pair<int, int>> block_indexs_;
    int block_cols_, block_rows_, block_cnts_;
    int track_max_block_features_;

    // 视差计算
    // Parallax in pixels
    double parallax_map_{0}, parallax_ref_{0};
    int parallax_map_counts_{0}, parallax_ref_counts_{0};

    bool isnewkeyframe_{false};
    bool isinitializing_{true};

    // 直方图统计值
    // Using histogram to detect and remove the frame with drastic illumination change
    double histogram_;
    int passed_cnt_{0};

    // Configurations
    double track_min_parallax_;
    int track_max_features_;
    int track_min_pixel_distance_;
    double track_max_interval_;
    bool track_check_histogram_;

    double reprojection_error_std_;

    bool is_use_visualization_;

    FileSaver::Ptr logfilesaver_;

    TimeCost timecost_;
    vector<double> logging_data_;

    //drt_vio_init add
    vector<drtSFMFeature> SFMConstruct_;
    TrackChoose trackchoose_;

    EPNPEstimator::Ptr epnpsolver_;
    // vector<Pose> tmp_pose_list_;
    std::optional<std::tuple<double, Pose, Eigen::Matrix<double, 6, 6>>> pose_cur_prior_;
    std::vector<uint32_t> removesfm_ids_;

    Pose nextPose_;
};

#endif // GVINS_TRACKING_H
