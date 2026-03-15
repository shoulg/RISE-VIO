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

#include "tracking.h"

#include "common/angle.h"
#include "common/logging.h"
#include "common/rotation.h"

#include <tbb/tbb.h>
#include <yaml-cpp/yaml.h>

Tracking::Tracking(Camera::Ptr camera, Map::Ptr map, Drawer::Ptr drawer, const string &configfile,
                   const string &outputpath)
    : frame_cur_(nullptr)
    , frame_ref_(nullptr)
    , camera_(std::move(camera))
    , map_(std::move(map))
    , drawer_(std::move(drawer))
    , isnewkeyframe_(false)
    , isinitializing_(true)
    , histogram_(0)
    , trackchoose_(TRACK_C_PREINITIALING) {

    logfilesaver_ = FileSaver::create(outputpath + "/tracking.txt", 3);
    if (!logfilesaver_->isOpen()) {
        LOGE << "Failed to open data file";
        return;
    }

    YAML::Node config;
    std::vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    track_check_histogram_ = config["track_check_histogram"].as<bool>();
    track_min_parallax_    = config["track_min_parallax"].as<double>();
    track_max_features_    = config["track_max_features"].as<int>();
    track_max_interval_    = config["track_max_interval"].as<double>();
    track_max_interval_ *= 0.95; // 错开整时间间隔

    is_use_visualization_   = config["is_use_visualization"].as<bool>();
    reprojection_error_std_ = config["reprojection_error_std"].as<double>();

    // 直方图均衡化
    clahe_ = cv::createCLAHE(3.0, cv::Size(21, 21));

    // 分块索引
    block_cols_ = static_cast<int>(lround(camera_->width() / TRACK_BLOCK_SIZE));
    block_rows_ = static_cast<int>(lround(camera_->height() / TRACK_BLOCK_SIZE));
    block_cnts_ = block_cols_ * block_rows_;

    int col, row;
    row = camera_->height() / block_rows_;
    col = camera_->width() / block_cols_;
    block_indexs_.emplace_back(std::make_pair(col, row));
    for (int i = 0; i < block_rows_; i++) {
        for (int j = 0; j < block_cols_; j++) {
            block_indexs_.emplace_back(std::make_pair(col * j, row * i));
        }
    }

    if (config["feature_density_scale"]) {
        feature_density_scale_ = config["feature_density_scale"].as<double>();
    }

    if (config["initial_deltatime"]) {
        initial_deltatime_ = config["initial_deltatime"].as<double>();
    }

    // 每个分块提取的角点数量
    track_max_block_features_ =
        static_cast<int>(lround(static_cast<double>(track_max_features_) / static_cast<double>(block_cnts_)));

    // 每个格子的提取特征数量平方面积为格子面积的 2/3
    track_min_pixel_distance_ = static_cast<int>(round(TRACK_BLOCK_SIZE / sqrt(track_max_block_features_ * feature_density_scale_)));

    std::cout << "track_min_pixel_distance_: " << track_min_pixel_distance_ << std::endl;

    double gnc_epnp_noise = 0.005;
    if (config["gnc_epnp_noise"]) {
        gnc_epnp_noise = config["gnc_epnp_noise"].as<double>();
    }

    if (config["triangulate_parallax"]) {
        triangulate_parallax_ = config["triangulate_parallax"].as<double>();
    }

    if (config["offset"]) {
        offset_ = config["offset"].as<double>();
    }

    epnpsolver_ = EPNPEstimator::create(gnc_epnp_noise);
    initializeLogFile("/sad/catkin_ws/ex_logs", "epnp_changes.txt");
    clearLogsFolder("/sad/catkin_ws/ex_logs/matched_points");
    clearLogsFolder("/sad/catkin_ws/ex_logs/matched_points_keyframe");
}

double Tracking::calculateHistigram(const Mat &image) {
    Mat histogram;
    int channels[]         = {0};
    int histsize           = 256;
    float range[]          = {0, 256};
    const float *histrange = {range};
    bool uniform = true, accumulate = false;

    cv::calcHist(&image, 1, channels, Mat(), histogram, 1, &histsize, &histrange, uniform, accumulate);

    double hist = 0;
    for (int k = 0; k < 256; k++) {
        hist += histogram.at<float>(k) * (float) k / 256.0;
    }
    hist /= (image.cols * image.rows);

    return hist;
}

bool Tracking::preprocessing(Frame::Ptr frame) {
    isnewkeyframe_ = false;

    // 彩色转灰度
    if (frame->image().channels() == 3) {
        cv::cvtColor(frame->image(), frame->image(), cv::COLOR_BGR2GRAY);
    }

    if (track_check_histogram_) {
        // 计算直方图参数
        double hist = calculateHistigram(frame->image());
        if (histogram_ != 0) {
            double rate = fabs((hist - histogram_) / histogram_);

            // 图像直方图变化比例大于10%, 则跳过当前帧
            if (rate > 0.1) {
                LOGW << "Histogram change too large at " << Logging::doubleData(frame->stamp()) << " with " << rate;
                passed_cnt_++;

                if (passed_cnt_ > 1) {
                    histogram_ = 0;
                }
                return false;
            }
        }
        histogram_ = hist;
    }

    frame_pre_ = frame_cur_;
    frame_cur_ = std::move(frame);

    // 直方图均衡化
    // clahe_->apply(frame_cur_->image(), frame_cur_->image());

    return true;
}

TrackState Tracking::track(Frame::Ptr frame) {
    // Tracking

    timecost_.restart();

    TrackState track_state = TRACK_PASSED;

    // 预处理
    if (!preprocessing(std::move(frame))) {
        return track_state;
    }

    //tmp change
    if (trackchoose_ == TRACK_C_PREINITIALING) {
        if (frame_ref_ == nullptr) {
            doResetTracking();

            frame_ref_ = frame_cur_;

            featuresDetection(frame_ref_, false);

            return TRACK_CHECK;
        }

        if (pts2d_ref_.empty()) {
            featuresDetection(frame_ref_, false);
        }

        // 从参考帧跟踪过来的特征点
        trackReferenceFrame_old();

        double deltatime;
        deltatime = frame_cur_->stamp() - frame_ref_->stamp();
        if ((deltatime < 0.03) || (deltatime <= initial_deltatime_ && parallax_ref_ < track_min_parallax_)) {//deltatime <= 0.22 && 
            showTracking();
            // LOGI << "Initialization tracking with parallax " << parallax_ref_;
            return TRACK_PASSED;
        }//parallax_ref_ < 0.5 * track_min_parallax_

        movelastkeyframe();

        // 初始化两帧都是关键帧
        frame_ref_->setKeyFrame(KEYFRAME_NORMAL);

        // 新关键帧, 地图更新, 数据转存
        makeNewFrame_old(KEYFRAME_NORMAL);
        last_keyframe_ = frame_cur_;

        track_state = TRACK_CHECK;
    } else if (trackchoose_ == TRACK_C_INITIALING) {
        // Initialization
        {
            if (pts2d_ref_.empty()) {
                featuresDetection(frame_ref_, false);
            } else {
                remove2dpoints();
                featuresDetection(frame_pre_, true);
            }
        }

        trackMappoint();

        // 从参考帧跟踪过来的特征点
        trackReferenceFrame();

        if (parallax_ref_ < track_min_parallax_) {
            showTracking();
            return TRACK_INITIALIZING;
        }

        LOGI << "Initialization tracking with parallax " << parallax_ref_;

        triangulation();

        if (!frame_cur_->numFeatures()) {
            makeNewFrame(KEYFRAME_NORMAL);

            showTracking();
            last_keyframe_ = frame_cur_;
            return TRACK_INITIALIZING;
        }

        // 初始化两帧都是关键帧
        frame_ref_->setKeyFrame(KEYFRAME_NORMAL);

        // 新关键帧, 地图更新, 数据转存
        makeNewFrame(KEYFRAME_NORMAL);
        last_keyframe_ = frame_cur_;

        isinitializing_ = false;
        trackchoose_ = TRACK_C_INITIALIZED;

        track_state = TRACK_TRACKING;
    } else if (trackchoose_ == TRACK_C_INITIALIZED) {
        // Tracking
        {        
            remove2dpoints();
            featuresDetection(frame_pre_, true);
        }

        // 跟踪上一帧中带路标点的特征, 利用预测的位姿先验
        trackMappoint();

        // 未关联路标点的新特征, 补偿旋转预测
        trackReferenceFrame();

        // 检查关键帧类型
        auto keyframe_state = checkKeyFrameSate();

        // 正常关键帧, 需要三角化路标点
        if ((keyframe_state == KEYFRAME_NORMAL) || (keyframe_state == KEYFRAME_REMOVE_OLDEST)) {
            // 三角化补充路标点
            triangulation();
            // auto pose_opt = checkgncpose();
            // if (pose_opt) {
            //     LOGI << "Use gnc pose.";
            //     triangulation(*pose_opt);
            // } else {
            //     triangulation();
            // }
        } else {
            // 添加新的特征
            featuresDetection(frame_cur_, true);
        }

        // 跟踪失败, 路标点数据严重不足
        if (doResetTracking()) {
            trackchoose_ = TRACK_C_INITIALING;
            makeNewFrame(KEYFRAME_NORMAL);
            return TRACK_LOST;
        }

        // 观测帧, 进行插入
        if (keyframe_state != KEYFRAME_NONE) {
            checkmappoint();
            makeNewFrame(keyframe_state);
        }

        track_state = TRACK_TRACKING;

        if (keyframe_state != KEYFRAME_NONE) {
            writeLoggingMessage();
        }
    }

    // 显示跟踪情况
    showTracking();

    return track_state;
}

bool Tracking::isGoodDepth(double depth, double scale) {
    return ((depth > MapPoint::NEAREST_DEPTH) && (depth < MapPoint::FARTHEST_DEPTH * scale));
}

void Tracking::makeNewFrame(int state) {
    frame_cur_->setKeyFrame(state);
    isnewkeyframe_ = true;

    //drt_vio_init add
    updateConstruct();

    // 仅当正常关键帧才更新参考帧
    if ((state == KEYFRAME_NORMAL) || (state == KEYFRAME_REMOVE_OLDEST)) {
        frame_ref_ = frame_cur_;

        featuresDetection(frame_ref_, true);
    }

}

keyFrameState Tracking::checkKeyFrameSate() {
    keyFrameState keyframe_state = KEYFRAME_NONE;

    // 相邻时间太短, 不进行关键帧处理
    double dt = frame_cur_->stamp() - last_keyframe_->stamp();
    if (dt < TRACK_MIN_INTERVAl) {
        return keyframe_state;
    }

    double parallax = (parallax_map_ * parallax_map_counts_ + parallax_ref_ * parallax_ref_counts_) /
                      (parallax_map_counts_ + parallax_ref_counts_);
    if (parallax > track_min_parallax_) {
        // 新的关键帧, 满足最小像素视差

        keyframe_state = map_->isWindowFull() ? KEYFRAME_REMOVE_OLDEST : KEYFRAME_NORMAL;

        LOGI << "Keyframe at " << Logging::doubleData(frame_cur_->stamp()) << ", mappoints "
             << frame_cur_->numFeatures() << ", interval " << dt << ", parallax " << parallax;
    } else if (dt > track_max_interval_) {
        // 普通观测帧, 非关键帧
        keyframe_state = KEYFRAME_REMOVE_SECOND_NEW;
        LOGI << "Keyframe at " << Logging::doubleData(frame_cur_->stamp()) << " due to long interval";
    }

    // 切换上一关键帧, 用于时间间隔计算
    if (keyframe_state != KEYFRAME_NONE) {
        last_keyframe_ = frame_cur_;

        // 更新路标点在观测帧中的使用次数
        for (auto &mappoint : tracked_mappoint_) {
            mappoint->increaseUsedTimes();
        }

        // 输出关键帧信息
        logging_data_.clear();

        logging_data_.push_back(frame_cur_->stamp());
        logging_data_.push_back(dt);
        logging_data_.push_back(parallax);
        logging_data_.push_back(relativeTranslation());
        logging_data_.push_back(relativeRotation());
    }

    return keyframe_state;
}

void Tracking::writeLoggingMessage() {
    logging_data_.push_back(static_cast<double>(frame_cur_->features().size()));
    logging_data_.push_back(timecost_.costInMillisecond());

    logfilesaver_->dump(logging_data_);
    logfilesaver_->flush();
}

bool Tracking::doResetTracking() {
    if (!frame_cur_->numFeatures()) {
        isinitializing_ = true;
        frame_ref_      = frame_cur_;
        pts2d_new_.clear();
        pts2d_ref_.clear();
        pts2d_ref_frame_.clear();
        velocity_ref_.clear();
        //drt_vio_init add
        SFMConstruct_.clear();
        return true;
    }

    return false;
}

double Tracking::relativeTranslation() {
    return (frame_cur_->pose().t - frame_ref_->pose().t).norm();
}

double Tracking::relativeRotation() {
    Matrix3d R     = frame_cur_->pose().R.transpose() * frame_ref_->pose().R;
    Vector3d euler = Rotation::matrix2euler(R);

    // Only for heading
    return fabs(euler[1] * R2D);
}

void Tracking::showTracking() {
    if (!is_use_visualization_) {
        return;
    }

    drawer_->updateFrame(frame_cur_);
}

bool Tracking::trackMappoint() {

    // 上一帧中的路标点
    mappoint_matched_.clear();
    vector<cv::Point2f> pts2d_map, pts2d_matched, pts2d_map_undis;
    //tmp change
    vector<Eigen::Vector3d> pts3d_map;

    vector<MapPointType> mappoint_type;
    auto features = frame_pre_->features();
    for (auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (mappoint && !mappoint->isOutlier()) {
            mappoint_matched_.push_back(mappoint);
            pts2d_map_undis.push_back(feature.second->keyPoint());
            pts2d_map.push_back(feature.second->distortedKeyPoint());
            mappoint_type.push_back(mappoint->mapPointType());

            // 预测的特征点
            auto pixel = camera_->world2pixel(mappoint->pos(), frame_cur_->pose());

            pts2d_matched.emplace_back(pixel);

            //tmp change
            pts3d_map.emplace_back(camera_->world2cam(mappoint->pos(), frame_pre_->pose()));
        }
    }
    if (pts2d_matched.empty()) {
        LOGE << "No feature with mappoint in previous frame";
        return false;
    }

    // 预测的特征点像素坐标添加畸变, 用于跟踪
    camera_->distortPoints(pts2d_matched);

    // MODIFIED: 在执行光流之前，保存一份预测点（用于后续 reprojection gating）
    // vector<cv::Point2f> pts2d_pred = pts2d_matched; // ADDED

    vector<uint8_t> status, status_reverse;
    vector<float> error;
    vector<cv::Point2f> pts2d_reverse = pts2d_map;

    // 正向光流
    cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_map, pts2d_matched, status, error,
                             cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    //tmp change
    int succ_num = 0;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
            succ_num++;
    }
    if (succ_num < 10)
        cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_map, pts2d_matched, status, error, cv::Size(21, 21), 3);

    // 亚像素精度优化
    // cv::cornerSubPix(
    //     frame_cur_->image(),              // 当前图像
    //     pts2d_matched,                    // 要优化的点
    //     cv::Size(3, 3),                   // 搜索窗口
    //     cv::Size(-1, -1),                 // 死区大小
    //     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01)
    // );

    // 反向光流
    cv::calcOpticalFlowPyrLK(frame_cur_->image(), frame_pre_->image(), pts2d_matched, pts2d_reverse, status_reverse,
                             error, cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);//TRACK_PYRAMID_LEVEL

    // 跟踪失败的
    size_t num_1 = 0, num_2 = 0, num_3 = 0, num_4 = 0;
    for (size_t k = 0; k < status.size(); k++) {
        if (!status[k]) num_1++;
        if (!status_reverse[k]) num_2++;
        if (isOnBorder(pts2d_matched[k])) num_3++;
        if (!(ptsDistance(pts2d_reverse[k], pts2d_map[k]) <= 0.8)) num_4++;

        if (status[k] && status_reverse[k] && !isOnBorder(pts2d_matched[k]) &&
            (ptsDistance(pts2d_reverse[k], pts2d_map[k]) <= 0.8)) {
            status[k] = 1;
        } else {
            status[k] = 0;
        }
    }
    LOGI << "Track:" << status.size() << ", Fail: " << num_1 << ", " << num_2 << ", " << num_3 << ", " << num_4 << ";";

    reduceVector(pts2d_map, status);
    //tmp change
    reduceVector(pts3d_map, status);

    // reduceVector(pts2d_pred, status);

    reduceVector(pts2d_matched, status);
    reduceVector(mappoint_matched_, status);
    reduceVector(mappoint_type, status);
    reduceVector(pts2d_map_undis, status);

    if (pts2d_matched.empty()) {
        LOGE << "Track previous with mappoint failed";
        // 清除上一帧的跟踪
        if (is_use_visualization_) {
            drawer_->updateTrackedMapPoints({}, {}, {});
        }
        parallax_map_        = 0;
        parallax_map_counts_ = 0;
        return false;
    }

    //tmp change
    vector<cv::Point2f> pts2d_matched_norm;
    pts2d_matched_norm.reserve(pts2d_matched.size());
    for (const auto &pt2d_matched: pts2d_matched) {
        Eigen::Vector3d cam3d = camera_->pixel2cam(camera_->undistortPointManual(pt2d_matched));
        pts2d_matched_norm.emplace_back(static_cast<float>(cam3d.x()), static_cast<float>(cam3d.y()));
    }

    vector<double> w_guess;
    size_t long_num = 0;
    for (const auto &mappoint: mappoint_matched_) {
        double w;
        if (mappoint->observedTimes() > 4) {//optimizedTimes()
            w = 1.0;
            long_num++;
        } else {
            w = 0.0;
        }
        w_guess.push_back(w);
    }

    if (long_num > 31 && w_guess.size() > 0 && frame_pre_->id() > scheduled_threshold_) {//long_num > 50
        //gnc epnp
        vector<Eigen::Vector3d> cws;
        epnpsolver_->reset(pts3d_map, w_guess, pts2d_matched_norm);
        Pose pji = camera_->posei2j(frame_pre_->pose(), frame_cur_->pose());
        epnpsolver_->gnc_estimate(pji.R, pji.t);
        pts2d_matched_norm = epnpsolver_->takePts2();
        status = epnpsolver_->takeBinaryStatus();
        Pose epnp_pose = epnpsolver_->takePose();
        // tmp_pose_list_.push_back(epnp_pose);
        cws = epnpsolver_->takecws();
        
        if (epnpsolver_->takeneedremove()) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(9) << frame_pre_->stamp();
            std::string filename = "/sad/catkin_ws/ex_logs/matched_points/" + oss.str() + ".txt";

            // 在合适的位置写入文件，比如 for 循环结束后：
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Failed to open file for writing." << std::endl;
            } else {
                //记录 frame_ref_ 的时间戳
                file << std::fixed << std::setprecision(9);  // 保留9位小数，可根据实际时间戳精度调整
                file << "# frame_ref_stamp: " << frame_pre_->stamp() << std::endl;

                for (size_t i = 0; i < 4; ++i) {
                    file << "cws " << i << ": " << cws[i][0] << " " << cws[i][1] << " " << cws[i][2] << std::endl;
                }

                //记录 IMU 估计的旋转矩阵 R 和 t 向量
                file << "# imu_rotation_matrix (3x3):" << std::endl;
                for (int i = 0; i < 3; ++i) {
                    file << pji.R.row(i) << std::endl;
                }
                file << "# imu_translation_vector (3x1):" << std::endl;
                file << pji.t.transpose() << std::endl;

                file << "# epnp_rotation_matrix (3x3):" << std::endl;
                for (int i = 0; i < 3; ++i) {
                    file << epnp_pose.R.row(i) << std::endl;
                }
                file << "# epnp_translation_vector (3x1):" << std::endl;
                file << epnp_pose.t.transpose() << std::endl;

                for (size_t i = 0; i < pts2d_matched_norm.size(); ++i) {
                    const auto& pt2d_map = pts2d_map[i];
                    const auto& pt2d_matched = pts2d_matched[i];
                    const auto& pt2d_matched_norm = pts2d_matched_norm[i];
                    const auto& pt3d_map = pts3d_map[i];
                    double w = w_guess[i];
                    float sta = status[i];
                    file << pt2d_map.x << " " << pt2d_map.y << " "
                        << pt2d_matched.x << " " << pt2d_matched.y << " "
                        << pt2d_matched_norm.x << " " << pt2d_matched_norm.y << " "
                        << pt3d_map.x() << " " << pt3d_map.y() << " " << pt3d_map.z() << " "
                        << w << " " << sta << std::endl;
                }

                file.close();
                std::cout << "Saved matched points to matched_points.txt" << std::endl;
            }
        }

        // 跟踪失败的
        size_t num_5 = 0;
        for (size_t k = 0; k < status.size(); k++) {
            if (!status[k]) num_5++;
        }
        LOGI << "GNC_EPNP Check:" << status.size() << ", Fail: " << num_5 <<  ";";

        double rot_deg, trans_error;
        camera_->computePoseError(pji, epnp_pose, rot_deg, trans_error);
        // LOGI << "TRACK Rot Error: " << rot_deg << ", Trans Error: " << trans_error << ";";
        insertLogLine("*** TRACK MAPPOINT ***", "/sad/catkin_ws/ex_logs", "epnp_changes.txt");
        // appendLog(frame_cur_->stamp(), rot_deg, trans_error, "/sad/catkin_ws/ex_logs", "epnp_changes.txt");

        reduceVector(pts2d_map, status);

        //tmp change
        reduceVector(pts3d_map, status);
        reduceVector(pts2d_matched_norm, status);

        reduceVector(pts2d_matched, status);
        reduceVector(mappoint_matched_, status);
        reduceVector(mappoint_type, status);
        reduceVector(pts2d_map_undis, status);

        // frame_cur_->setPose(camera_->posej2w(frame_pre_->pose(), epnp_pose));
    // } else {
    //     const double reproj_thresh = 4.0; // 像素阈值，可根据分辨率/体验调节（建议 3~8）
    //     size_t reproj_out = 0;
    //     status.assign(pts2d_matched.size(), 1);

    //     // 注意：此处 pts2d_pred 与 pts2d_matched 的索引是一一对应的（未做 reduce 前）
    //     for (size_t k = 0; k < pts2d_matched.size(); ++k) {
    //         double err = cv::norm(pts2d_matched[k] - pts2d_pred[k]); // 像素误差
    //         if (err > reproj_thresh) {
    //             status[k] = 0; // 标记为失败
    //             reproj_out++;
    //         }
    //     }

    //     LOGI << "Reproj-gating removed: " << reproj_out << " / " << pts2d_matched.size() << " points";

    //     // 使用现有的 reduceVector 管道剔除这些点（和之前保持一致的索引机制）
    //     reduceVector(pts2d_map, status);
    //     reduceVector(pts3d_map, status);

    //     reduceVector(pts2d_matched, status);
    //     reduceVector(mappoint_matched_, status);
    //     reduceVector(mappoint_type, status);
    //     reduceVector(pts2d_map_undis, status);

    //     if (pts2d_matched.empty()) {
    //         LOGE << "Track previous with mappoint failed after reproj gating";
    //         if (is_use_visualization_) {
    //             drawer_->updateTrackedMapPoints({}, {}, {});
    //         }
    //         parallax_map_ = 0;
    //         parallax_map_counts_ = 0;
    //         return false;
    //     }
    }

    // 匹配后的点, 需要重新矫正畸变
    auto pts2d_matched_undis = pts2d_matched;
    camera_->undistortPoints(pts2d_matched_undis);

    // 匹配的3D-2D
    frame_cur_->clearFeatures();
    tracked_mappoint_.clear();

    double dt = frame_cur_->stamp() - frame_pre_->stamp();
    for (size_t k = 0; k < pts2d_matched_undis.size(); k++) {
        auto mappoint = mappoint_matched_[k];
        Vector3d pc = camera_->pixel2cam(pts2d_matched_undis[k]);

        // 将3D-2D匹配到的landmarks指向到当前帧
        auto velocity = (camera_->pixel2cam(pts2d_matched_undis[k]) - camera_->pixel2cam(pts2d_map_undis[k])) / dt;
        auto feature  = Feature::createFeature(frame_cur_, {velocity.x(), velocity.y()}, pc, pts2d_matched_undis[k],
                                               pts2d_matched[k], FEATURE_MATCHED, frame_cur_->stamp());
        mappoint->addObservation(frame_cur_->stamp(), feature);
        feature->addMapPoint(mappoint);
        frame_cur_->addFeature(mappoint->id(), feature);

        // 用于更新使用次数
        tracked_mappoint_.push_back(mappoint);
    }

    // 路标点跟踪情况
    if (is_use_visualization_) {
        drawer_->updateTrackedMapPoints(pts2d_map, pts2d_matched, mappoint_type);
    }

    parallax_map_counts_ = parallaxFromReferenceMapPoints(parallax_map_);

    LOGI << "Track " << tracked_mappoint_.size() << " map points";

    return true;
}

bool Tracking::trackReferenceFrame() {

    if (pts2d_ref_.empty()) {
        LOGW << "No new feature in previous frame " << Logging::doubleData(frame_cur_->stamp());
        return false;
    }

    // 补偿旋转预测
    Matrix3d r_cur_pre = frame_cur_->pose().R.transpose() * frame_pre_->pose().R;

    // 原始畸变补偿
    auto pts2d_new_undis = pts2d_new_;
    camera_->undistortPoints(pts2d_new_undis);

    pts2d_cur_.clear();
    for (const auto &pp_pre : pts2d_new_undis) {
        Vector3d pc_pre = camera_->pixel2cam(pp_pre);
        Vector3d pc_cur = r_cur_pre * pc_pre;

        // 添加畸变
        auto pp_cur = camera_->distortCameraPoint(pc_cur);
        pts2d_cur_.emplace_back(pp_cur);
    }

    // 跟踪参考帧
    vector<uint8_t> status, status_reverse;
    vector<float> error;
    vector<cv::Point2f> pts2d_reverse = pts2d_new_;

    // 正向光流
    cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_new_, pts2d_cur_, status, error,
                             cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    // pts2d_cur_.clear();
    // cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_new_, pts2d_cur_, status,  error,
    //                          cv::Size(21, 21), 3);

    // 亚像素精度优化
    cv::cornerSubPix(
        frame_cur_->image(),              // 当前图像
        pts2d_cur_,                    // 要优化的点
        cv::Size(3, 3),                   // 搜索窗口
        cv::Size(-1, -1),                 // 死区大小
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01)
    );

    // 反向光流
    cv::calcOpticalFlowPyrLK(frame_cur_->image(), frame_pre_->image(), pts2d_cur_, pts2d_reverse, status_reverse, error,
                             cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    // 剔除跟踪失败的, 正向反向跟踪在0.5个像素以内
    for (size_t k = 0; k < status.size(); k++) {
        if (status[k] && status_reverse[k] && !isOnBorder(pts2d_cur_[k]) &&
            (ptsDistance(pts2d_reverse[k], pts2d_new_[k]) <= 0.5)) {
            status[k] = 1;
        } else {
            status[k] = 0;
        }
    }
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(pts2d_new_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(velocity_ref_, status);
    //drt_vio_init add
    reduceVector(SFMConstruct_, status);

    if (pts2d_ref_.empty()) {
        LOGW << "No new feature in previous frame";
        drawer_->updateTrackedRefPoints({}, {});
        return false;
    }

    // 原始带畸变的角点
    pts2d_new_undis      = pts2d_new_;
    auto pts2d_cur_undis = pts2d_cur_;

    camera_->undistortPoints(pts2d_new_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算像素速度
    velocity_cur_.clear();
    double dt = frame_cur_->stamp() - frame_pre_->stamp();

    for (size_t k = 0; k < pts2d_cur_undis.size(); k++) {
        Vector3d vel      = (camera_->pixel2cam(pts2d_cur_undis[k]) - camera_->pixel2cam(pts2d_new_undis[k])) / dt;
        Vector2d velocity = {vel.x(), vel.y()};
        velocity_cur_.push_back(velocity);

        // 在关键帧后新增加的特征
        if (pts2d_ref_frame_[k]->id() > frame_ref_->id()) {
            velocity_ref_[k] = velocity;
        }
    }

    // 计算视差
    auto pts2d_ref_undis = pts2d_ref_;
    camera_->undistortPoints(pts2d_ref_undis);
    parallax_ref_counts_ = parallaxFromReferenceKeyPoints(pts2d_ref_undis, pts2d_cur_undis, parallax_ref_);

    // Fundamental粗差剔除
    if (pts2d_cur_.size() >= 15) {
        cv::findFundamentalMat(pts2d_new_undis, pts2d_cur_undis, cv::FM_RANSAC, 1.5, 0.99, status);//reprojection_error_std_

        reduceVector(pts2d_ref_, status);
        reduceVector(pts2d_cur_, status);
        reduceVector(pts2d_ref_frame_, status);
        reduceVector(velocity_cur_, status);
        reduceVector(velocity_ref_, status);
        //drt_vio_init add
        reduceVector(SFMConstruct_, status);
    }

    if (pts2d_cur_.empty()) {
        LOGW << "No new feature in previous frame";
        drawer_->updateTrackedRefPoints({}, {});
        return false;
    }

    // 从参考帧跟踪过来的新特征点
    if (is_use_visualization_) {
        drawer_->updateTrackedRefPoints(pts2d_ref_, pts2d_cur_);
    }

    // 用于下一帧的跟踪
    pts2d_new_ = pts2d_cur_;
    //drt_vio_init add
    addConstructobs(pts2d_cur_);

    LOGI << "Track " << pts2d_new_.size() << " reference points";

    return !pts2d_new_.empty();
}

void Tracking::featuresDetection(Frame::Ptr &frame, bool ismask) {

    // 特征点足够则无需提取
    int num_features = static_cast<int>(frame->features().size() + pts2d_ref_.size());
    if (num_features > (track_max_features_ - 5)) {
        return;
    }

    // 初始化分配内存
    int features_cnts[block_cnts_];
    vector<vector<cv::Point2f>> block_features(block_cnts_);
    // 必要的分配内存, 否则并行会造成数据结构错乱
    for (auto &block : block_features) {
        block.reserve(track_max_block_features_);
    }
    for (int k = 0; k < block_cnts_; k++) {
        features_cnts[k] = 0;
    }

    // 计算每个分块已有特征点数量
    int col, row;
    for (const auto &feature : frame->features()) {
        col = int(feature.second->keyPoint().x / (float) block_indexs_[0].first);
        row = int(feature.second->keyPoint().y / (float) block_indexs_[0].second);
        features_cnts[row * block_cols_ + col]++;
    }
    for (auto &pts2d : pts2d_new_) {
        col = int(pts2d.x / (float) block_indexs_[0].first);
        row = int(pts2d.y / (float) block_indexs_[0].second);
        features_cnts[row * block_cols_ + col]++;
    }

    // 设置感兴趣区域, 没有特征的区域
    Mat mask = Mat(camera_->size(), CV_8UC1, 255);
    if (ismask) {
        // 已经跟踪上的点
        for (const auto &pt : frame_cur_->features()) {
            cv::circle(mask, pt.second->keyPoint(), track_min_pixel_distance_, 0, cv::FILLED);
        }

        // 还在跟踪的点
        for (const auto &pts2d : pts2d_new_) {
            cv::circle(mask, pts2d, track_min_pixel_distance_, 0, cv::FILLED);
        }
    }

    // 亚像素角点提取参数
    cv::Size win_size          = cv::Size(3, 3);
    cv::Size zero_zone         = cv::Size(-1, -1);
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    auto tracking_function = [&](const tbb::blocked_range<int> &range) {
        for (int k = range.begin(); k != range.end(); k++) {
            int blocl_track_num = track_max_block_features_ - features_cnts[k];
            if (blocl_track_num > 0) {

                int cols = k % block_cols_;
                int rows = k / block_cols_;

                int col_sta = cols * block_indexs_[0].first;
                int col_end = col_sta + block_indexs_[0].first;
                int row_sta = rows * block_indexs_[0].second;
                int row_end = row_sta + block_indexs_[0].second;
                if (k != (block_cnts_ - 1)) {
                    col_end -= 2;
                    row_end -= 2;
                }

                Mat block_image = frame->image().colRange(col_sta, col_end).rowRange(row_sta, row_end);
                Mat block_mask  = mask.colRange(col_sta, col_end).rowRange(row_sta, row_end);

                cv::goodFeaturesToTrack(block_image, block_features[k], blocl_track_num, 0.01,
                                        track_min_pixel_distance_, block_mask);
                if (!block_features[k].empty()) {
                    // 获取亚像素角点
                    cv::cornerSubPix(block_image, block_features[k], win_size, zero_zone, term_crit);
                }
            }
        }
    };
    tbb::parallel_for(tbb::blocked_range<int>(0, block_cnts_), tracking_function);

    // 调整角点的坐标
    int num_new_features = 0;

    // 连续跟踪的角点, 未三角化的点
    if (!ismask) {
        pts2d_new_.clear();
        pts2d_ref_.clear();
        pts2d_ref_frame_.clear();
        velocity_ref_.clear();
        //drt_vio_init add
        SFMConstruct_.clear();
    }

    for (int k = 0; k < block_cnts_; k++) {
        col = k % block_cols_;
        row = k / block_cols_;

        for (const auto &point : block_features[k]) {
            float x = static_cast<float>(col * block_indexs_[0].first) + point.x;
            float y = static_cast<float>(row * block_indexs_[0].second) + point.y;

            auto pts2d = cv::Point2f(x, y);
            pts2d_ref_.push_back(pts2d);
            pts2d_new_.push_back(pts2d);
            pts2d_ref_frame_.push_back(frame);
            velocity_ref_.emplace_back(0, 0);
            //drt_vio_init add
            auto pts2d_undis = camera_->undistortPointManual(pts2d);
            auto pt = camera_->pixel2cam(pts2d_undis);
            double time = frame->stamp();
            SFMConstruct_.emplace_back(time, FeaturePerFrame(time, pt, pts2d_undis, pts2d, Vector2d(0, 0)));

            num_new_features++;
        }
    }

    LOGI << "Add " << num_new_features << " new features to " << num_features;
}

bool Tracking::triangulation() {
    // 无跟踪上的特征
    if (pts2d_cur_.empty()) {
        return false;
    }

    Pose pose0;
    Pose pose1 = frame_cur_->pose();

    Eigen::Matrix<double, 3, 4> T_c_w_0, T_c_w_1;
    T_c_w_1 = pose2Tcw(pose1).topRows<3>();

    int num_succeeded = 0;
    int num_outlier   = 0;
    int num_reset     = 0;
    int num_outtime   = 0;

    // 原始带畸变的角点
    std::vector<cv::Point2f> pts2d_ref_undis = pts2d_ref_;
    std::vector<cv::Point2f> pts2d_cur_undis = pts2d_cur_;

    // 矫正畸变以进行三角化
    camera_->undistortPoints(pts2d_ref_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算使用齐次坐标, 相机坐标系
    vector<uint8_t> status;
    status.reserve(pts2d_cur_.size());
    for (size_t k = 0; k < pts2d_cur_.size(); k++) {
        auto pp0 = pts2d_ref_undis[k];
        auto pp1 = pts2d_cur_undis[k];

        // 参考帧
        auto frame_ref = pts2d_ref_frame_[k];
        if (frame_ref->id() > frame_ref_->id()) {
            // 中途添加的特征, 修改参考帧
            pts2d_ref_frame_[k] = frame_cur_;
            pts2d_ref_[k]       = pts2d_cur_[k];
            status.push_back(1);
            num_reset++;
            continue;
        }

        // 移除长时间跟踪导致参考帧已经不在窗口内的观测
        if (map_->isWindowNormal() && !map_->isValidKeyFrame(frame_ref)) {
            status.push_back(0);
            num_outtime++;
            continue;
        }

        // 进行必要的视差检查, 保证三角化有效
        pose0           = frame_ref->pose();
        double parallax = keyPointParallax(pts2d_ref_undis[k], pts2d_cur_undis[k], pose0, pose1);
        if (parallax < triangulate_parallax_) {
            status.push_back(1);
            continue;
        }

        T_c_w_0 = pose2Tcw(pose0).topRows<3>();

        // 三角化
        Vector3d pc0 = camera_->pixel2cam(pts2d_ref_undis[k]);
        Vector3d pc1 = camera_->pixel2cam(pts2d_cur_undis[k]);
        Vector3d pw;
        triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, pw);

        // 三角化错误的点剔除
        //tmp change
        if (!isGoodToTrack(pp0, pose0, pw, 1.2, 3.0) || !isGoodToTrack(pp1, pose1, pw, 1.2, 3.0)) {//1.5 3.0
            //tmp change
            status.push_back(0);
            num_outlier++;
            continue;
        }
        status.push_back(0);
        num_succeeded++;

        // 新的路标点, 加入新的观测, 路标点加入地图
        auto pc       = camera_->world2cam(pw, frame_ref->pose());
        double depth  = pc.z();
        auto mappoint = MapPoint::createMapPoint(frame_ref, pw, pts2d_ref_undis[k], depth, MAPPOINT_TRIANGULATED);

        auto feature = Feature::createFeature(frame_cur_, velocity_cur_[k], pc1, pts2d_cur_undis[k], pts2d_cur_[k],
                                              FEATURE_TRIANGULATED, frame_cur_->stamp());
        mappoint->addObservation(frame_cur_->stamp(), feature);
        feature->addMapPoint(mappoint);
        frame_cur_->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        feature = Feature::createFeature(frame_ref, velocity_ref_[k], pc0, pts2d_ref_undis[k], pts2d_ref_[k],
                                         FEATURE_TRIANGULATED, frame_ref->stamp());
        mappoint->addObservation(frame_ref->stamp(), feature);
        feature->addMapPoint(mappoint);
        frame_ref->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        // 新三角化的路标点缓存到最新的关键帧, 不直接加入地图
        frame_cur_->addNewUnupdatedMappoint(mappoint);
    }

    updateremoveid(status);

    // 由于视差不够未及时三角化的角点
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(velocity_ref_, status);
    //drt_vio_init add
    reduceVector(SFMConstruct_, status);

    pts2d_new_ = pts2d_cur_;

    LOGI << "Triangulate " << num_succeeded << " 3D points with " << pts2d_cur_.size() << " left, " << num_reset
         << " reset, " << num_outtime << " outtime and " << num_outlier << " outliers";
    return true;
}

void Tracking::triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                                const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw) {
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();

    design_matrix.row(0) = pc0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = pc0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = pc1[0] * pose1.row(2) - pose1.row(0);
    design_matrix.row(3) = pc1[1] * pose1.row(2) - pose1.row(1);

    Eigen::Vector4d point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    pw                    = point.head<3>() / point(3);
}

bool Tracking::isGoodToTrack(const cv::Point2f &pp, const Pose &pose, const Vector3d &pw, double scale,
                             double depth_scale) {
    // 当前相机坐标系
    Vector3d pc = camera_->world2cam(pw, pose);

    // 深度检查
    if (!isGoodDepth(pc[2], depth_scale)) {
        return false;
    }

    // 重投影误差检查
    if (camera_->reprojectionError(pose, pw, pp).norm() > reprojection_error_std_ * scale) {
        return false;
    }

    return true;
}

template <typename T> void Tracking::reduceVector(T &vec, const vector<uint8_t> &status) {
    size_t index = 0;
    for (size_t k = 0; k < vec.size(); k++) {
        if (status[k]) {
            vec[index++] = vec[k];
        }
    }
    vec.resize(index);
}

//drt_vio_init add
void Tracking::addConstructobs(const vector<cv::Point2f> &pts2d) {
    double time = frame_cur_->stamp();
    for (size_t k = 0; k < pts2d.size(); k++) {
        auto &SFMFeature = SFMConstruct_[k];
        auto pts2d_undis = camera_->undistortPointManual(pts2d[k]);
        auto point = camera_->pixel2cam(pts2d_undis);
        SFMFeature.obs.emplace(time, FeaturePerFrame(time, point, pts2d_undis, pts2d[k], velocity_cur_[k]));
    }
}

void Tracking::updateConstruct() {
    double ref_time = frame_ref_->stamp();
    double cur_time = frame_cur_->stamp();

    // Lambda：移除观测时间在 (ref_time, cur_time) 之间的 obs
    auto remove_invalid_obs = [ref_time, cur_time, this](std::vector<drtSFMFeature>& construct) {
        // size_t idx = 0;
        for (auto& feature : construct) {
            for (auto it = feature.obs.begin(); it != feature.obs.end(); ) {
                Eigen::Vector3d point3d = it->second.point();  // 得到3D点
                assert(point3d(2) != 0.0 && "Invalid 3D point: z = 0");
                if (ref_time < it->first && it->first < cur_time) {
                    it = feature.obs.erase(it);
                // } else if (it->first == cur_time) {
                //     it->second.setvelocity(velocity_cur_[idx]);
                //     ++it;
                // } else if (it->first == ref_time) {
                //     it->second.setvelocity(velocity_ref_[idx]);
                //     ++it;
                } else {
                    ++it;
                }
            }
            // idx++;
        }
    };

    remove_invalid_obs(SFMConstruct_);
    map_->updateSFMConstruct(cur_time, SFMConstruct_);
    if (removesfm_ids_.size() > 0) {
        map_->removesfmfeaturebyid(removesfm_ids_);
    }
}

bool Tracking::isOnBorder(const cv::Point2f &pts) {
    return pts.x < 3.0 || pts.y < 3.0 || (pts.x > (camera_->width() - 3.0)) || (pts.y > (camera_->height() - 3.0));
}

Eigen::Matrix4d Tracking::pose2Tcw(const Pose &pose) {
    Eigen::Matrix4d Tcw;
    Tcw.setZero();
    Tcw(3, 3) = 1;

    Tcw.block<3, 3>(0, 0) = pose.R.transpose();
    Tcw.block<3, 1>(0, 3) = -pose.R.transpose() * pose.t;
    return Tcw;
}

double Tracking::keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0,
                                  const Pose &pose1) {
    Vector3d pc0 = camera_->pixel2cam(pp0);
    Vector3d pc1 = camera_->pixel2cam(pp1);

    // 补偿掉旋转
    Vector3d pc01 = pose1.R.transpose() * pose0.R * pc0;

    // 像素大小
    return (pc01.head<2>() - pc1.head<2>()).norm() * camera_->focalLength();
}

int Tracking::parallaxFromReferenceMapPoints(double &parallax) {

    parallax      = 0;
    int counts    = 0;
    auto features = frame_ref_->features();

    for (auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (mappoint && !mappoint->isOutlier()) {
            // 取最新的一个路标点观测
            auto observations = mappoint->observations();
            if (observations.empty()) {
                continue;
            }
            auto feat = observations.at(mappoint->end_time()).lock();
            if (feat && !feat->isOutlier()) {
                auto frame = feat->getFrame();
                if (frame && (frame == frame_cur_)) {
                    // 对应同一路标点在当前帧的像素观测
                    parallax += keyPointParallax(feature.second->keyPoint(), feat->keyPoint(), frame_ref_->pose(),
                                                 frame_cur_->pose());
                    counts++;
                }
            }
        }
    }

    if (counts != 0) {
        parallax /= counts;
    }

    return counts;
}

int Tracking::parallaxFromReferenceKeyPoints(const vector<cv::Point2f> &ref, const vector<cv::Point2f> &cur,
                                             double &parallax) {
    parallax   = 0;
    int counts = 0;
    for (size_t k = 0; k < pts2d_ref_frame_.size(); k++) {
        if (pts2d_ref_frame_[k] == frame_ref_) {
            parallax += keyPointParallax(ref[k], cur[k], frame_ref_->pose(), frame_cur_->pose());
            counts++;
        }
    }
    if (counts != 0) {
        parallax /= counts;
    }

    return counts;
}

bool Tracking::trackReferenceFrame_old() {
    if (pts2d_ref_.empty()) {
        LOGW << "No new feature in previous frame " << Logging::doubleData(frame_cur_->stamp());
        return false;
    }

    // 补偿旋转预测
    Matrix3d r_cur_pre = frame_cur_->pose().R.transpose() * frame_pre_->pose().R;

    // 原始畸变补偿
    auto pts2d_new_undis = pts2d_new_;
    camera_->undistortPoints(pts2d_new_undis);

    pts2d_cur_.clear();
    for (const auto &pp_pre : pts2d_new_undis) {
        Vector3d pc_pre = camera_->pixel2cam(pp_pre);
        Vector3d pc_cur = r_cur_pre * pc_pre;

        // 添加畸变
        auto pp_cur = camera_->distortCameraPoint(pc_cur);
        pts2d_cur_.emplace_back(pp_cur);
    }

    // 跟踪参考帧
    vector<uint8_t> status, status_reverse;
    vector<float> error;
    vector<cv::Point2f> pts2d_reverse = pts2d_new_;

    //正向光流
    // cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_new_, pts2d_cur_, status,  error,
    //                          cv::Size(21, 21), 3);

    cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_new_, pts2d_cur_, status, error,
                             cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    //tmp change
    int succ_num = 0;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
            succ_num++;
    }
    if (succ_num < 30)
        cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_new_, pts2d_cur_, status, error, cv::Size(21, 21), 3);

    // 亚像素精度优化
    cv::cornerSubPix(
        frame_cur_->image(),              // 当前图像
        pts2d_cur_,                    // 要优化的点
        cv::Size(3, 3),                   // 搜索窗口
        cv::Size(-1, -1),                 // 死区大小
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01)
    );

    // 反向光流
    cv::calcOpticalFlowPyrLK(frame_cur_->image(), frame_pre_->image(), pts2d_cur_, pts2d_reverse, status_reverse, error,
                             cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);//TRACK_PYRAMID_LEVEL

    // 剔除跟踪失败的, 正向反向跟踪在0.5个像素以内
    for (size_t k = 0; k < status.size(); k++) {
        if (status[k] && status_reverse[k] && !isOnBorder(pts2d_cur_[k]) &&
            (ptsDistance(pts2d_reverse[k], pts2d_new_[k]) <= 0.5)) {
            status[k] = 1;
        } else {
            status[k] = 0;
        }
    }
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(pts2d_new_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(velocity_ref_, status);
    //drt_vio_init add
    reduceVector(SFMConstruct_, status);

    if (pts2d_ref_.empty()) {
        LOGW << "No new feature in previous frame";
        drawer_->updateTrackedRefPoints({}, {});
        return false;
    }

    // 原始畸变补偿
    pts2d_new_undis = pts2d_new_;
    auto pts2d_cur_undis = pts2d_cur_;

    camera_->undistortPoints(pts2d_new_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算像素速度
    velocity_cur_.clear();
    double dt = frame_cur_->stamp() - frame_pre_->stamp();

    for (size_t k = 0; k < pts2d_cur_undis.size(); k++) {
        Vector3d vel      = (camera_->pixel2cam(pts2d_cur_undis[k]) - camera_->pixel2cam(pts2d_new_undis[k])) / dt;
        Vector2d velocity = {vel.x(), vel.y()};
        velocity_cur_.push_back(velocity);

        // 在关键帧后新增加的特征
        if (pts2d_ref_frame_[k]->id() > frame_ref_->id()) {
            velocity_ref_[k] = velocity;
        }
    }

    // 计算视差
    auto pts2d_ref_undis = pts2d_ref_;
    camera_->undistortPoints(pts2d_ref_undis);
    parallax_ref_counts_ = parallaxFromReferenceKeyPoints(pts2d_ref_undis, pts2d_cur_undis, parallax_ref_);

    LOGI << "Parallax From Reference KeyPoints: " << parallax_ref_ << ", with Parallax Counts: " << parallax_ref_counts_;

    // Fundamental粗差剔除
    if (pts2d_cur_.size() >= 15) {
        cv::findFundamentalMat(pts2d_new_undis, pts2d_cur_undis, cv::FM_RANSAC, 1.5, 0.99, status);//reprojection_error_std_

        reduceVector(pts2d_ref_, status);
        reduceVector(pts2d_cur_, status);
        reduceVector(pts2d_ref_frame_, status);
        reduceVector(velocity_cur_, status);
        reduceVector(velocity_ref_, status);
        //drt_vio_init add
        reduceVector(SFMConstruct_, status);
    }

    if (pts2d_cur_.empty()) {
        LOGW << "No new feature in previous frame";
        drawer_->updateTrackedRefPoints({}, {});
        return false;
    }

    // 从参考帧跟踪过来的新特征点
    if (is_use_visualization_) {
        drawer_->updateTrackedRefPoints(pts2d_ref_, pts2d_cur_);
    }

    // 用于下一帧的跟踪
    pts2d_new_ = pts2d_cur_;
    //drt_vio_init add
    addConstructobs(pts2d_cur_);

    LOGI << "Track " << pts2d_new_.size() << " reference points";

    return !pts2d_new_.empty();
}

void Tracking::makeNewFrame_old(int state) {
    frame_cur_->setKeyFrame(state);

    //drt_vio_init add
    updateConstruct();

    // 仅当正常关键帧才更新参考帧
    if ((state == KEYFRAME_NORMAL) || (state == KEYFRAME_REMOVE_OLDEST)) {
        frame_ppre_ = frame_ref_;
        frame_ref_ = frame_cur_;

        featuresDetection(frame_ref_, true);
    }
}

bool Tracking::movelastkeyframe() {
    // 无跟踪上的特征
    if (pts2d_cur_.empty()) {
        return false;
    }

    // 原始带畸变的角点
    auto pts2d_ref_dis = pts2d_ref_;
    auto pts2d_cur_dis = pts2d_cur_;

    // 计算使用齐次坐标, 相机坐标系
    vector<uint8_t> status;
    status.reserve(pts2d_cur_.size());
    for (size_t k = 0; k < pts2d_cur_.size(); k++) {
        // auto pp0 = pts2d_ref_dis[k];
        // auto pp1 = pts2d_cur_dis[k];

        // 参考帧
        auto frame_ref = pts2d_ref_frame_[k];
        if (frame_ref->id() > frame_ref_->id()) {
            // 中途添加的特征, 修改参考帧
            pts2d_ref_frame_[k] = frame_cur_;
            pts2d_ref_[k]       = pts2d_cur_[k];
            status.push_back(1);
            continue;
        }

        // 移除长时间跟踪导致参考帧已经不在窗口内的观测
        if (map_->isWindowNormal() && !map_->isKeyFrameInMap(frame_ref)) {
            status.push_back(0);
            continue;
        }

        status.push_back(1);

        // pts2d_ref_frame_[k] = frame_cur_;
        // pts2d_ref_[k]       = pts2d_cur_[k];
        // status.push_back(1);
    }

    // 由于视差不够未及时三角化的角点
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(velocity_ref_, status);
    //drt_vio_init add
    reduceVector(SFMConstruct_, status);

    pts2d_new_ = pts2d_cur_;

    LOGI << "Move Last Keyframe && Just keep track 2d features.";
    return true;
}

// void Tracking::checkmappoint() {
//     vector<cv::Point2f> pts2d_matched_norm;
//     vector<Eigen::Vector3d> pts3d_map;
//     vector<cv::Point2f> pts2d_map, pts2d_matched;
//     vector<double> w_guess;
//     size_t long_num = 0;
//     for (auto &mappoint: mappoint_matched_) {
//         auto observations = mappoint->observations();
//         if (observations.find(frame_ref_->stamp()) != observations.end() &&
//             observations.find(frame_cur_->stamp()) != observations.end()) {
//             auto feature_cur = observations.at(frame_cur_->stamp()).lock();
//             pts2d_matched_norm.emplace_back(feature_cur->point_2d());
//             pts2d_matched.emplace_back(feature_cur->distortedKeyPoint());
//             auto feature_ref = observations.at(frame_ref_->stamp()).lock();
//             pts2d_map.emplace_back(feature_ref->distortedKeyPoint());
//         } else {
//             continue;
//         }
//         double w;
//         if (mappoint->observedTimes() > 5) {//optimizedTimes()
//             w = 1.0;
//             long_num++;
//         } else {
//             w = 0.0;
//         }
//         w_guess.push_back(w);
//         pts3d_map.emplace_back(camera_->world2cam(mappoint->pos(), frame_ref_->pose()));
//     }
//     if (long_num > 31 && w_guess.size() > 0) {//long_num > 50
//         //gnc epnp
//         vector<uint8_t> status;
//         vector<Eigen::Vector3d> cws;
//         epnpsolver_->reset(pts3d_map, w_guess, pts2d_matched_norm);
//         Pose pji = camera_->posei2j(frame_ref_->pose(), frame_cur_->pose());
//         epnpsolver_->gnc_estimate(pji.R, pji.t);
//         pts2d_matched_norm = epnpsolver_->takePts2();
//         status = epnpsolver_->takeBinaryStatus();
//         Pose epnp_pose = epnpsolver_->takePose();
//         // tmp_pose_list_.push_back(epnp_pose);
//         cws = epnpsolver_->takecws();
//         std::ostringstream oss;
//         oss << std::fixed << std::setprecision(9) << frame_ref_->stamp();
//         std::string filename = "/sad/catkin_ws/ex_logs/matched_points_keyframe/" + oss.str() + ".txt";
//         // 在合适的位置写入文件，比如 for 循环结束后：
//         std::ofstream file(filename);
//         if (!file.is_open()) {
//             std::cerr << "Failed to open file for writing." << std::endl;
//         } else {
//             //记录 frame_ref_ 的时间戳
//             file << std::fixed << std::setprecision(9);  // 保留9位小数，可根据实际时间戳精度调整
//             file << "# frame_ref_stamp: " << frame_ref_->stamp() << std::endl;
//             for (size_t i = 0; i < 4; ++i) {
//                 file << "cws " << i << ": " << cws[i][0] << " " << cws[i][1] << " " << cws[i][2] << std::endl;
//             }
//             //记录 IMU 估计的旋转矩阵 R 和 t 向量
//             file << "# imu_rotation_matrix (3x3):" << std::endl;
//             for (int i = 0; i < 3; ++i) {
//                 file << pji.R.row(i) << std::endl;
//             }
//             file << "# imu_translation_vector (3x1):" << std::endl;
//             file << pji.t.transpose() << std::endl;
//             file << "# epnp_rotation_matrix (3x3):" << std::endl;
//             for (int i = 0; i < 3; ++i) {
//                 file << epnp_pose.R.row(i) << std::endl;
//             }
//             file << "# epnp_translation_vector (3x1):" << std::endl;
//             file << epnp_pose.t.transpose() << std::endl;
//             for (size_t i = 0; i < pts2d_matched_norm.size(); ++i) {
//                 const auto& pt2d_map = pts2d_map[i];
//                 const auto& pt2d_matched = pts2d_matched[i];
//                 const auto& pt2d_matched_norm = pts2d_matched_norm[i];
//                 const auto& pt3d_map = pts3d_map[i];
//                 double w = w_guess[i];
//                 float sta = status[i];
//                 file << pt2d_map.x << " " << pt2d_map.y << " "
//                     << pt2d_matched.x << " " << pt2d_matched.y << " "
//                     << pt2d_matched_norm.x << " " << pt2d_matched_norm.y << " "
//                     << pt3d_map.x() << " " << pt3d_map.y() << " " << pt3d_map.z() << " "
//                     << w << " " << sta << std::endl;
//             }
//             file.close();
//             std::cout << "Saved matched points to matched_points.txt" << std::endl;
//         }
//         // 跟踪失败的
//         size_t num_5 = 0;
//         for (size_t k = 0; k < status.size(); k++) {
//             if (!status[k]) num_5++;
//         }
//         LOGI << "GNC_EPNP Check:" << status.size() << ", Fail: " << num_5 <<  " at " << std::fixed << std::setprecision(9) << frame_cur_->stamp() << ";";
//         double rot_deg, trans_error;
//         camera_->computePoseError(pji, epnp_pose, rot_deg, trans_error);
//         LOGI << "CHECK Rot Error: " << rot_deg << ", Trans Error: " << trans_error << ";";
//         insertLogLine("*** CHECK MAPPOINT ***", "/sad/catkin_ws/ex_logs", "epnp_changes.txt");
//         appendLog(frame_cur_->stamp(), rot_deg, trans_error, "/sad/catkin_ws/ex_logs", "epnp_changes.txt");
//         // Eigen::Matrix<double, 6, 6> visualsqrt_info_ = epnpsolver_->computeSqrtInfoFromVisualPrior((status.size() - num_5), rot_deg, trans_error);
//         // pose_cur_prior_ = std::make_tuple(frame_cur_->stamp(), camera_->posei2k(epnp_pose, frame_ref_->pose()), visualsqrt_info_);
//         for (size_t i = 0; i < status.size(); i++) {
//             if (!status[i]) {
//                 mappoint_matched_[i]->setOutlier(true);
//             }
//         }
//     } else {
//         LOGI << "GNC_EPNP has not enough points;";
//     }
// }

void Tracking::checkmappoint() {
    cv::Mat img_ref = frame_ref_->rawImage(); // <- 替换为实际的 getter
    cv::Mat img_cur = frame_cur_->rawImage(); // <- 替换为实际的 getter
    cv::Mat harris;
    cv::cornerHarris(img_ref, harris, 2, 3, 0.04);
    cv::Mat harris_norm;
    cv::normalize(harris, harris_norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    // 若需要局部梯度能量（备用），可预计算 Sobel 能量
    cv::Mat Ix, Iy;
    cv::Sobel(img_ref, Ix, CV_32F, 1, 0, 3);
    cv::Sobel(img_ref, Iy, CV_32F, 0, 1, 3);
    cv::Mat grad_energy;
    cv::magnitude(Ix, Iy, grad_energy);
    cv::normalize(grad_energy, grad_energy, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    vector<cv::Point2f> pts2d_matched_norm;
    vector<Eigen::Vector3d> pts3d_map;
    vector<uint8_t> status_more;
    vector<cv::Point2f> pts2d_map, pts2d_matched;
    vector<double> w_guess;
    size_t long_num = 0;
    for (auto &mappoint: mappoint_matched_) {
        auto observations = mappoint->observations();
        if (observations.find(frame_ref_->stamp()) != observations.end() &&
            observations.find(frame_cur_->stamp()) != observations.end()) {
            auto feature_cur = observations.at(frame_cur_->stamp()).lock();
            pts2d_matched_norm.emplace_back(feature_cur->point_2d());
            pts2d_matched.emplace_back(feature_cur->distortedKeyPoint());
            auto feature_ref = observations.at(frame_ref_->stamp()).lock();
            pts2d_map.emplace_back(feature_ref->distortedKeyPoint());
        } else {
            continue;
        }
        pts3d_map.emplace_back(camera_->world2cam(mappoint->pos(), frame_ref_->pose()));
        auto feature_cur = observations.at(frame_cur_->stamp()).lock();
        auto feature_ref = observations.at(frame_ref_->stamp()).lock();
        // --- 1) 关键点响应 / 角点强度（从 Harris 图读取） ---
        float resp_val = 0.0f;
        cv::Point2f p_ref_px = feature_ref->distortedKeyPoint();
        int px = static_cast<int>(std::round(p_ref_px.x));
        int py = static_cast<int>(std::round(p_ref_px.y));
        if (px >= 0 && px < harris_norm.cols && py >= 0 && py < harris_norm.rows) {
            resp_val = harris_norm.at<float>(py, px); // 归一化到 0..1
        } else {
            // 若越界，则尝试用局部梯度能量作为替代
            if (px >= 0 && px < grad_energy.cols && py >= 0 && py < grad_energy.rows)
                resp_val = grad_energy.at<float>(py, px);
        }
        // --- 2) IMU 投影一致性（像素级重投影误差） ---
        // 将 pt3d_ref 用 pji 变换到 cur 相机系并投影
        double imu_reproj_err_pix = camera_->reprojectionError(frame_cur_->pose(), mappoint->pos(), feature_cur->keyPoint()).norm();
        status_more.push_back(imu_reproj_err_pix < 4.0);
        double imu_proj_score = std::max(0.0, 1.0 - imu_reproj_err_pix / 2.0); // 0..1
        // --- 归一化 observedTimes 得分 ---
        int obs_times = mappoint->observedTimes();
        double obs_score = double(obs_times) / double(obs_times + 5); // 0..1
        // 计算初始权重
        double w = 0.2 * resp_val
                + 0.45 * imu_proj_score
                + 0.35 * obs_score;
        if (w > 0.5) long_num++;
        w_guess.push_back(w);
    }
    if (long_num > 31 && w_guess.size() > 0 && frame_ref_->id() > scheduled_threshold_) {//long_num > 50
        //gnc epnp
        vector<uint8_t> status;
        vector<Eigen::Vector3d> cws;
        epnpsolver_->reset(pts3d_map, w_guess, pts2d_matched_norm);
        Pose pji = camera_->posei2j(frame_ref_->pose(), frame_cur_->pose());
        epnpsolver_->gnc_estimate(pji.R, pji.t);
        pts2d_matched_norm = epnpsolver_->takePts2();
        status = epnpsolver_->takeBinaryStatus();
        Pose epnp_pose = epnpsolver_->takePose();
        // tmp_pose_list_.push_back(epnp_pose);
        cws = epnpsolver_->takecws();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9) << frame_ref_->stamp();
        std::string filename = "/sad/catkin_ws/ex_logs/matched_points_keyframe/" + oss.str() + ".txt";
        // 在合适的位置写入文件，比如 for 循环结束后：
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
        } else {
            //记录 frame_ref_ 的时间戳
            file << std::fixed << std::setprecision(9);  // 保留9位小数，可根据实际时间戳精度调整
            file << "# frame_ref_stamp: " << frame_ref_->stamp() << std::endl;
            for (size_t i = 0; i < 4; ++i) {
                file << "cws " << i << ": " << cws[i][0] << " " << cws[i][1] << " " << cws[i][2] << std::endl;
            }
            //记录 IMU 估计的旋转矩阵 R 和 t 向量
            file << "# imu_rotation_matrix (3x3):" << std::endl;
            for (int i = 0; i < 3; ++i) {
                file << pji.R.row(i) << std::endl;
            }
            file << "# imu_translation_vector (3x1):" << std::endl;
            file << pji.t.transpose() << std::endl;
            file << "# epnp_rotation_matrix (3x3):" << std::endl;
            for (int i = 0; i < 3; ++i) {
                file << epnp_pose.R.row(i) << std::endl;
            }
            file << "# epnp_translation_vector (3x1):" << std::endl;
            file << epnp_pose.t.transpose() << std::endl;
            for (size_t i = 0; i < pts2d_matched_norm.size(); ++i) {
                const auto& pt2d_map = pts2d_map[i];
                const auto& pt2d_matched = pts2d_matched[i];
                const auto& pt2d_matched_norm = pts2d_matched_norm[i];
                const auto& pt3d_map = pts3d_map[i];
                double w = w_guess[i];
                float sta = status[i];
                file << pt2d_map.x << " " << pt2d_map.y << " "
                    << pt2d_matched.x << " " << pt2d_matched.y << " "
                    << pt2d_matched_norm.x << " " << pt2d_matched_norm.y << " "
                    << pt3d_map.x() << " " << pt3d_map.y() << " " << pt3d_map.z() << " "
                    << w << " " << sta << std::endl;
            }
            file.close();
            std::cout << "Saved matched points to matched_points.txt" << std::endl;
        }
        // 跟踪失败的
        size_t num_5 = 0;
        for (size_t k = 0; k < status.size(); k++) {
            if (!status[k]) num_5++;
        }
        LOGI << "GNC_EPNP Check:" << status.size() << ", Fail: " << num_5 <<  " at " << std::fixed << std::setprecision(9) << frame_cur_->stamp() << ";";
        double rot_deg, trans_error;
        camera_->computePoseError(pji, epnp_pose, rot_deg, trans_error);
        // LOGI << "CHECK Rot Error: " << rot_deg << ", Trans Error: " << trans_error << ";";
        insertLogLine("*** CHECK MAPPOINT ***", "/sad/catkin_ws/ex_logs", "epnp_changes.txt");
        // appendLog(frame_cur_->stamp(), rot_deg, trans_error, "/sad/catkin_ws/ex_logs", "epnp_changes.txt");
        // Eigen::Matrix<double, 6, 6> visualsqrt_info_ = epnpsolver_->computeSqrtInfoFromVisualPrior((status.size() - num_5), rot_deg, trans_error);
        // pose_cur_prior_ = std::make_tuple(frame_cur_->stamp(), camera_->posei2k(epnp_pose, frame_ref_->pose()), visualsqrt_info_);
        for (size_t i = 0; i < status.size(); i++) {
            if (!status[i]) {
                mappoint_matched_[i]->setOutlier(true);
            }
        }
    } else {
        size_t num_5 = 0;
        for (size_t i = 0; i < status_more.size(); i++) {
            if (!status_more[i]) {
                mappoint_matched_[i]->setOutlier(true);
                num_5++;
            }
        }
        LOGI << "GNC_EPNP has not enough points, with " << num_5 << " points to remove;";
    }
}

std::optional<Pose> Tracking::checkgncpose() {
    vector<cv::Point2f> pts2d_matched_norm;
    vector<Eigen::Vector3d> pts3d_map;
    vector<cv::Point2f> pts2d_map, pts2d_matched;
    vector<double> w_guess;
    size_t long_num = 0;
    for (auto &mappoint: mappoint_matched_) {
        auto observations = mappoint->observations();
        if (observations.find(frame_ref_->stamp()) != observations.end() &&
            observations.find(frame_cur_->stamp()) != observations.end()) {
            auto feature_cur = observations.at(frame_cur_->stamp()).lock();
            pts2d_matched_norm.emplace_back(feature_cur->point_2d());
            pts2d_matched.emplace_back(feature_cur->distortedKeyPoint());

            auto feature_ref = observations.at(frame_ref_->stamp()).lock();
            pts2d_map.emplace_back(feature_ref->distortedKeyPoint());
        } else {
            continue;
        }

        double w;
        if (mappoint->optimizedTimes() > 1) {
            w = 1.0;
            long_num++;
        // } else if (mappoint->optimizedTimes() >= 4 && mappoint->optimizedTimes() <= 5) {
        //     w = 0.1;
        } else {
            w = 0.0;
        }
        w_guess.push_back(w);

        pts3d_map.emplace_back(camera_->world2cam(mappoint->pos(), frame_ref_->pose()));
    }

    if (long_num > 48 && w_guess.size() > 0) {//long_num > 50
        //gnc epnp
        vector<float> status;
        vector<Eigen::Vector3d> cws;
        epnpsolver_->reset(pts3d_map, w_guess, pts2d_matched_norm);
        Pose pji = camera_->posei2j(frame_ref_->pose(), frame_cur_->pose());
        epnpsolver_->gnc_estimate(pji.R, pji.t);
        pts2d_matched_norm = epnpsolver_->takePts2();
        status = epnpsolver_->takeStatus();
        Pose epnp_pose = epnpsolver_->takePose();
        // tmp_pose_list_.push_back(epnp_pose);
        cws = epnpsolver_->takecws();

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9) << frame_pre_->stamp();
        std::string filename = "/sad/catkin_ws/ex_logs/matched_points_keyframe/" + oss.str() + ".txt";

        // 在合适的位置写入文件，比如 for 循环结束后：
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
        } else {
            //记录 frame_ref_ 的时间戳
            file << std::fixed << std::setprecision(9);  // 保留9位小数，可根据实际时间戳精度调整
            file << "# frame_ref_stamp: " << frame_ref_->stamp() << std::endl;

            for (size_t i = 0; i < 4; ++i) {
                file << "cws " << i << ": " << cws[i][0] << " " << cws[i][1] << " " << cws[i][2] << std::endl;
            }

            //记录 IMU 估计的旋转矩阵 R 和 t 向量
            file << "# imu_rotation_matrix (3x3):" << std::endl;
            for (int i = 0; i < 3; ++i) {
                file << pji.R.row(i) << std::endl;
            }
            file << "# imu_translation_vector (3x1):" << std::endl;
            file << pji.t.transpose() << std::endl;

            file << "# epnp_rotation_matrix (3x3):" << std::endl;
            for (int i = 0; i < 3; ++i) {
                file << epnp_pose.R.row(i) << std::endl;
            }
            file << "# epnp_translation_vector (3x1):" << std::endl;
            file << epnp_pose.t.transpose() << std::endl;

            for (size_t i = 0; i < pts2d_matched_norm.size(); ++i) {
                const auto& pt2d_map = pts2d_map[i];
                const auto& pt2d_matched = pts2d_matched[i];
                const auto& pt2d_matched_norm = pts2d_matched_norm[i];
                const auto& pt3d_map = pts3d_map[i];
                double w = w_guess[i];
                float sta = status[i];
                file << pt2d_map.x << " " << pt2d_map.y << " "
                    << pt2d_matched.x << " " << pt2d_matched.y << " "
                    << pt2d_matched_norm.x << " " << pt2d_matched_norm.y << " "
                    << pt3d_map.x() << " " << pt3d_map.y() << " " << pt3d_map.z() << " "
                    << w << " " << sta << std::endl;
            }

            file.close();
            std::cout << "Saved matched points to matched_points.txt" << std::endl;
        }

        // 跟踪失败的
        size_t num_5 = 0;
        for (size_t k = 0; k < status.size(); k++) {
            if (!status[k]) num_5++;
        }
        LOGI << "GNC_EPNP Check:" << status.size() << ", Fail: " << num_5 <<  " at " << std::fixed << std::setprecision(9) << frame_cur_->stamp() << ";";

        double rot_deg, trans_error;
        camera_->computePoseError(pji, epnp_pose, rot_deg, trans_error);
        LOGI << "CHECK Rot Error: " << rot_deg << ", Trans Error: " << trans_error << ";";
        insertLogLine("*** CHECK MAPPOINT ***", "/sad/catkin_ws/ex_logs", "epnp_changes.txt");
        appendLog(frame_cur_->stamp(), rot_deg, trans_error, "/sad/catkin_ws/ex_logs", "epnp_changes.txt");

        for (size_t i = 0; i < status.size(); i++) {
            if (!status[i]) {
                mappoint_matched_[i]->setOutlier(true);
            }
        }

        return camera_->posej2w(frame_ref_->pose(), epnp_pose);
    } else {
        LOGI << "GNC_EPNP has not enough points;";
        return std::nullopt;
    }
}

bool Tracking::triangulation(const Pose &gncpose) {
    // 无跟踪上的特征
    if (pts2d_cur_.empty()) {
        return false;
    }

    Pose pose0;
    Pose pose1 = gncpose;

    Eigen::Matrix<double, 3, 4> T_c_w_0, T_c_w_1;
    T_c_w_1 = pose2Tcw(pose1).topRows<3>();

    int num_succeeded = 0;
    int num_outlier   = 0;
    int num_reset     = 0;
    int num_outtime   = 0;

    // 原始带畸变的角点
    auto pts2d_ref_undis = pts2d_ref_;
    auto pts2d_cur_undis = pts2d_cur_;

    // 矫正畸变以进行三角化
    camera_->undistortPoints(pts2d_ref_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算使用齐次坐标, 相机坐标系
    vector<uint8_t> status;
    status.reserve(pts2d_cur_.size());
    for (size_t k = 0; k < pts2d_cur_.size(); k++) {
        auto pp0 = pts2d_ref_undis[k];
        auto pp1 = pts2d_cur_undis[k];

        // 参考帧
        auto frame_ref = pts2d_ref_frame_[k];
        if (frame_ref->id() > frame_ref_->id()) {
            // 中途添加的特征, 修改参考帧
            pts2d_ref_frame_[k] = frame_cur_;
            pts2d_ref_[k]       = pts2d_cur_[k];
            status.push_back(1);
            num_reset++;
            continue;
        }

        // 移除长时间跟踪导致参考帧已经不在窗口内的观测
        if (map_->isWindowNormal() && !map_->isValidKeyFrame(frame_ref)) {
            status.push_back(0);
            num_outtime++;
            continue;
        }

        // 进行必要的视差检查, 保证三角化有效
        pose0           = frame_ref->pose();
        double parallax = keyPointParallax(pts2d_ref_undis[k], pts2d_cur_undis[k], pose0, pose1);
        if (parallax < 15) {
            status.push_back(1);
            continue;
        }

        T_c_w_0 = pose2Tcw(pose0).topRows<3>();

        // 三角化
        Vector3d pc0 = camera_->pixel2cam(pts2d_ref_undis[k]);
        Vector3d pc1 = camera_->pixel2cam(pts2d_cur_undis[k]);
        Vector3d pw;
        triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, pw);

        // 三角化错误的点剔除
        //tmp change
        if (!isGoodToTrack(pp0, pose0, pw, 1.5, 3.0) || !isGoodToTrack(pp1, pose1, pw, 1.5, 3.0)) {//1.5 3.0
            //tmp change
            status.push_back(0);
            num_outlier++;
            continue;
        }
        status.push_back(0);
        num_succeeded++;

        // 新的路标点, 加入新的观测, 路标点加入地图
        auto pc       = camera_->world2cam(pw, frame_ref->pose());
        double depth  = pc.z();
        auto mappoint = MapPoint::createMapPoint(frame_ref, pw, pts2d_ref_undis[k], depth, MAPPOINT_TRIANGULATED);

        auto feature = Feature::createFeature(frame_cur_, velocity_cur_[k], pc1, pts2d_cur_undis[k], pts2d_cur_[k],
                                              FEATURE_TRIANGULATED, frame_cur_->stamp());
        mappoint->addObservation(frame_cur_->stamp(), feature);
        feature->addMapPoint(mappoint);
        frame_cur_->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        feature = Feature::createFeature(frame_ref, velocity_ref_[k], pc0, pts2d_ref_undis[k], pts2d_ref_[k],
                                         FEATURE_TRIANGULATED, frame_ref->stamp());
        mappoint->addObservation(frame_ref->stamp(), feature);
        feature->addMapPoint(mappoint);
        frame_ref->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        // 新三角化的路标点缓存到最新的关键帧, 不直接加入地图
        frame_cur_->addNewUnupdatedMappoint(mappoint);
    }

    updateremoveid(status);

    // 由于视差不够未及时三角化的角点
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(velocity_ref_, status);
    //drt_vio_init add
    reduceVector(SFMConstruct_, status);

    pts2d_new_ = pts2d_cur_;

    LOGI << "Triangulate " << num_succeeded << " 3D points with " << pts2d_cur_.size() << " left, " << num_reset
         << " reset, " << num_outtime << " outtime and " << num_outlier << " outliers";
    return true;
}

void Tracking::predictNextPose() {
    // Retrieve previous poses and timestamps
    Pose pose1 = frame_ppre_->pose();
    Pose pose2 = frame_pre_->pose();
    double t1 = frame_ppre_->stamp();
    double t2 = frame_pre_->stamp();
    double t3 = frame_cur_->stamp();

    // Compute time intervals
    double dt1 = t2 - t1;
    double dt2 = t3 - t2;
    double scale = dt2 / dt1;

    // Extract positions
    Eigen::Vector3d P1 = pose1.t;
    Eigen::Vector3d P2 = pose2.t;
    
    // Linear velocity and position prediction
    Eigen::Vector3d velocity = (P2 - P1) / dt1;
    Eigen::Vector3d P3 = P2 + velocity * dt2;

    // Extract rotations and convert to quaternions
    Eigen::Quaterniond Q1(pose1.R);
    Eigen::Quaterniond Q2(pose2.R);

    // Compute incremental rotation
    Eigen::Quaterniond dQ = Q1.conjugate() * Q2;

    // Helper to raise quaternion to a power
    auto quat_pow = [](const Eigen::Quaterniond& q, double t) {
        Eigen::AngleAxisd aa(q);
        return Eigen::Quaterniond(Eigen::AngleAxisd(aa.angle() * t, aa.axis()));
    };

    // Scaled incremental rotation and orientation prediction
    Eigen::Quaterniond dQ_scaled = quat_pow(dQ, scale);
    Eigen::Quaterniond Q3 = Q2 * dQ_scaled;

    // Write back predicted pose
    nextPose_.t = P3;
    nextPose_.R = Q3.toRotationMatrix();

    frame_ppre_ = frame_pre_;
}

void Tracking::remove2dpoints() {
    auto features = frame_pre_->features();
    std::vector<cv::Point2f> existing_pts;
    existing_pts.reserve(features.size());
    for (const auto &[id, feat] : features) {
        existing_pts.push_back(feat->distortedKeyPoint());
    }

    std::vector<uint8_t> status;
    status.reserve(pts2d_new_.size());

    for (const auto &pt_new : pts2d_new_) {
        bool is_far = true;
        for (const auto &pt_exist : existing_pts) {
            if (ptsDistance(pt_exist, pt_new) <= 0.5) {
                is_far = false;
                break;  // 提前退出
            }
        }
        status.push_back(is_far ? 1 : 0);
    }

    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(pts2d_new_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(velocity_ref_, status);
    //drt_vio_init add
    reduceVector(SFMConstruct_, status);
}

void Tracking::updateremoveid(const std::vector<unsigned char> &status) {
    removesfm_ids_.clear();
    removesfm_ids_.reserve(status.size());
    for (size_t i = 0; i < status.size(); i++) {
        auto sfmid = SFMConstruct_[i].id;
        if (status[i] == 0) {
            removesfm_ids_.push_back(sfmid);
        }
    }
}

void Tracking::gettracknum() {
    size_t num_3d = 0, num_2d = 0, num_all = 0;
    double pstamp = frame_pre_->stamp();
    if (frame_cur_ != nullptr) {
        num_3d = frame_cur_->features().size();
    }
    num_2d = pts2d_new_.size();
    num_all = num_3d + num_2d;
    LOGI << "frame_pre_: " << std::fixed << std::setprecision(9) << pstamp << ", num_3d: " << num_3d
         << ", num_2d: " << num_2d << ", num_all: " << num_all;
}
