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

#include "rb_vins.h"
#include "misc.h"

#include "common/angle.h"
#include "common/earth.h"
#include "common/gpstime.h"
#include "common/logging.h"

#include "factors/gnss_factor.h"
#include "factors/marginalization_factor.h"
#include "factors/marginalization_info.h"
#include "factors/pose_parameterization.h"
#include "factors/reprojection_factor.h"
#include "factors/residual_block_info.h"
#include "preintegration/imu_error_factor.h"
#include "preintegration/imu_mix_prior_factor.h"
#include "preintegration/imu_pose_prior_factor.h"
#include "preintegration/preintegration.h"
#include "preintegration/preintegration_factor.h"

#include "factors/reprojection_factor_w.h"
#include "morefactors/optimization.hpp"
#include "morefactors/visualprior_factor.h"
#include "morefuncs/filefuncs.h"

#include <ceres/ceres.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

using tbb::parallel_for;
using tbb::blocked_range;

#include <yaml-cpp/yaml.h>

#define TS(tag) do { \
    auto now = std::chrono::steady_clock::now(); \
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count(); \
    std::ostringstream oss; \
    oss << "[" << tag << "] " << std::fixed << ms << " ms"; \
    LOGI << oss.str(); \
} while(0)

GVINS::GVINS(const string &configfile, const string &outputpath, Drawer::Ptr drawer) {
    gvinsstate_ = GVINS_ERROR;

    initializeLogFile("/sad/catkin_ws/ex_logs", "biasg_changes.txt");
    initializeLogFile("/sad/catkin_ws/ex_logs", "map_point_rpe.txt");

    // 加载配置
    // Load configuration
    YAML::Node config;
    std::vector<double> vecdata;
    try {
        config = YAML::LoadFile(configfile);
    } catch (YAML::Exception &exception) {
        std::cout << "Failed to open configuration file" << std::endl;
        return;
    }

    // 文件IO
    // Output files
    ptsfilesaver_    = FileSaver::create(outputpath + "/mappoint.txt", 3);
    statfilesaver_   = FileSaver::create(outputpath + "/statistics.txt", 3);
    extfilesaver_    = FileSaver::create(outputpath + "/extrinsic.txt", 3);
    imuerrfilesaver_ = FileSaver::create(outputpath + "/IMU_ERR.bin", 7, FileSaver::BINARY);
    trajfilesaver_   = FileSaver::create(outputpath + "/trajectory.csv", 8);

    if (!ptsfilesaver_->isOpen() || !statfilesaver_->isOpen() || !extfilesaver_->isOpen()) {
        LOGE << "Failed to open data file";
        return;
    }

    // Make a copy of configuration file to the output directory
    std::ofstream ofconfig(outputpath + "/gvins.yaml");
    ofconfig << YAML::Dump(config);
    ofconfig.close();

    initlength_       = config["initlength"].as<int>();
    imudatarate_      = config["imudatarate"].as<double>();
    imudatadt_        = 1.0 / imudatarate_;
    reserved_ins_num_ = 2;

    // 安装参数
    // Installation parameters
    vecdata   = config["antlever"].as<std::vector<double>>();
    antlever_ = Vector3d(vecdata.data());

    // IMU噪声参数
    // IMU parameters
    integration_parameters_               = std::make_shared<IntegrationParameters>();
    integration_parameters_->gyr_arw      = config["imumodel"]["arw"].as<double>() * D2R / 60.0;
    integration_parameters_->gyr_bias_std = config["imumodel"]["gbstd"].as<double>() * D2R / 3600.0;
    integration_parameters_->acc_vrw      = config["imumodel"]["vrw"].as<double>() / 60.0;
    integration_parameters_->acc_bias_std = config["imumodel"]["abstd"].as<double>() * 1.0e-5;
    integration_parameters_->corr_time    = config["imumodel"]["corrtime"].as<double>() * 3600;
    integration_parameters_->gravity      = NORMAL_GRAVITY;

    integration_config_.iswithearth = config["iswithearth"].as<bool>();
    integration_config_.isuseodo    = false;
    integration_config_.iswithscale = false;
    integration_config_.gravity     = {0, 0, integration_parameters_->gravity};

    // 初始值, 后续根据GNSS定位实时更新
    // GNSS variables intializaiton
    integration_config_.origin.setZero();

    preintegration_options_ = Preintegration::getOptions(integration_config_);

    // 相机参数
    // Camera parameters
    vector<double> intrinsic  = config["cam0"]["intrinsic"].as<std::vector<double>>();
    vector<double> distortion = config["cam0"]["distortion"].as<std::vector<double>>();
    vector<int> resolution    = config["cam0"]["resolution"].as<std::vector<int>>();

    if (config["cam0"]["xi"]) {
        double xi = config["cam0"]["xi"].as<double>();
        camera_ = MEICamera::createCamera(intrinsic, distortion, xi, resolution);
    } else {
        camera_ = Camera::createCamera(intrinsic, distortion, resolution);
    }
    // IMU和Camera外参
    // Extrinsic parameters
    vecdata           = config["cam0"]["q_b_c"].as<std::vector<double>>();
    Quaterniond q_b_c = Eigen::Quaterniond(vecdata.data());
    vecdata           = config["cam0"]["t_b_c"].as<std::vector<double>>();
    Vector3d t_b_c    = Eigen::Vector3d(vecdata.data());
    td_b_c_           = config["cam0"]["td_b_c"].as<double>();

    pose_b_c_.R = q_b_c.toRotationMatrix();
    pose_b_c_.t = t_b_c;

    // 优化参数
    // Optimization parameters
    reprojection_error_std_      = config["reprojection_error_std"].as<double>();
    optimize_estimate_extrinsic_ = config["optimize_estimate_extrinsic"].as<bool>();
    optimize_estimate_td_        = config["optimize_estimate_td"].as<bool>();
    optimize_num_iterations_     = config["optimize_num_iterations"].as<int>();
    optimize_windows_size_       = config["optimize_windows_size"].as<size_t>();

    if (config["min_parallax_spd"]) {
        min_parallax_spd_     = config["min_parallax_spd"].as<double>();
        max_good_parallax_count_       = config["max_good_parallax_count"].as<double>();
    }

    if (config["noise_bound_sq"]) {
        noise_bound_sq_ = config["noise_bound_sq"].as<double>();
    }

    if (config["reprojection_error_std_scale"]) {
        reprojection_error_std_scale_ = config["reprojection_error_std_scale"].as<double>();
    }

    // 归一化相机坐标系下
    // Reprojection std
    optimize_reprojection_error_std_ = reprojection_error_std_scale_ * reprojection_error_std_ / camera_->focalLength();
    std::cout << "Focal length: " << camera_->focalLength() << ", with sqrt_info: " << 1.0 / optimize_reprojection_error_std_ << std::endl;

    // 可视化
    is_use_visualization_ = config["is_use_visualization"].as<bool>();

    // Initialize the containers
    preintegrationlist_.clear();
    statedatalist_.clear();
    timelist_.clear();

    //tmp change
    sub_preintegrationlist_.clear();
    sub_statedatalist_.clear();
    sub_timelist_.clear();

    int initial_triangulate_parallax = 15;
    if (config["initial_triangulate_parallax"]) {
        initial_triangulate_parallax = config["initial_triangulate_parallax"].as<double>();
    }

    // GVINS fusion objects
    map_    = std::make_shared<Map>(optimize_windows_size_, initial_triangulate_parallax);
    map_->setCamera(camera_);
    drawer_ = std::move(drawer);
    drawer_->setMap(map_);
    if (is_use_visualization_) {
        drawer_thread_ = std::thread(&Drawer::run, drawer_);
    }
    tracking_ = std::make_shared<Tracking>(camera_, map_, drawer_, configfile, outputpath);

    // Process threads
    fusion_thread_       = std::thread(&GVINS::runFusion, this);
    tracking_thread_     = std::thread(&GVINS::runTracking, this);
    optimization_thread_ = std::thread(&GVINS::runOptimization, this);

    gvinsstate_ = GVINS_INITIALIZING;
}

bool GVINS::addNewImu(const IMU &imu) {
    // TS("start imu input");
    if (imu_buffer_mutex_.try_lock()) {
        // bool was_empty = imu_buffer_.empty();
        if (imu.dt > (imudatadt_ * 1.5)) {
            LOGE << absl::StrFormat("Lost IMU data with at %0.3lf dt %0.3lf", imu.time, imu.dt);

            long cnts = lround(imu.dt / imudatadt_) - 1;

            IMU imudata  = imu;
            imudata.time = imu.time - imu.dt;
            while (cnts--) {
                imudata.time += imudatadt_;
                imudata.dt = imudatadt_;
                imu_buffer_.push(imudata);
                LOGE << "Append extra IMU data at " << Logging::doubleData(imudata.time);
            }
        } else {
            imu_buffer_.push(imu);
        }

        // 释放信号量
        // Release fusion semaphore
        // if (was_empty) {
            fusion_sem_.notify_one();
        // }

        imu_buffer_mutex_.unlock();
        // TS("imu input success");
        return true;
    } else {
        TS("imu input failed by imu_buffer_mutex_ lock failed");
    }

    return false;
}

bool GVINS::addNewFrame(const Frame::Ptr &frame) {
    // 只对第一个 frame 做特殊检查
    if (!first_frame_pushed_.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lk(ins_mutex_);
        if (ins_window_.empty()) {
            // 还没开始有任何经过 INS 机械编排的 IMU，等待下次再试
            return false;
        }
        // 拿 ins_window_ 中最早的 IMU 时间戳
        double imu0_time = ins_window_.front().first.time;
        if (imu0_time >= frame->stamp()) {
            // IMU 的最早时间不早于这帧图像 => 丢弃这帧图像
            return true;  
        }
        // 否则：至少有一个 imu.time < frame.time，可以放行第一个 frame
    }

    // 正常入队逻辑（除了第一个 frame 之外，之后都走这里）
    if (frame_buffer_mutex_.try_lock()) {
        frame_buffer_.push(frame);
        first_frame_pushed_.store(true, std::memory_order_relaxed);
        tracking_sem_.notify_one();
        frame_buffer_mutex_.unlock();
        return true;
    }
    // 竞争失败，GVINS 忙，让调用者保留这帧，下次再试
    return false;
}

// void GVINS::runFusion() {
//     IMU imu_pre, imu_cur;
//     IntegrationState state;
//     Frame::Ptr frame;
//     LOGI << "Fusion thread is started";
//     while (!isfinished_) { // While
//         Lock lock(fusion_mutex_);
//         fusion_sem_.wait(lock);
//         // 获取所有有效数据
//         // Process all IMU data
//         while (!imu_buffer_.empty()) { // IMU BUFFER
//             // 读取IMU缓存
//             // Load an IMU sample
//             {
//                 Lock lock2(imu_buffer_mutex_);
//                 imu_pre = imu_cur;
//                 imu_cur = imu_buffer_.front();
//                 imu_buffer_.pop();
//             }
//             // INS机械编排及INS处理
//             // INS mechanization
//             { // INS
//                 Lock lock3(ins_mutex_);
//                 if (!ins_window_.empty()) {
//                     // 上一时刻的状态
//                     // The INS state in last time for mechanization
//                     state = ins_window_.back().second;
//                 }
//                 ins_window_.emplace_back(imu_cur, IntegrationState());
//                 // 初始化完成后开始积分输出
//                 if (gvinsstate_ > GVINS_INITIALIZING) {
//                     if (isoptimized_ && state_mutex_.try_lock()) {
//                         // 优化求解结束, 需要更新IMU误差重新积分
//                         // When the optimization is finished
//                         isoptimized_ = false;
//                         state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
//                         // MISC::redoInsMechanization(integration_config_, state, reserved_ins_num_, ins_window_);
//                         MISC::redoInsMechanization_new(integration_config_, state, reserved_ins_num_, ins_window_);
//                         // tmp change
//                         LOGI << "bias: " << state.bg[0] << " " << state.bg[1] << " " << state.bg[2] << " " << state.ba[0] << " " << state.ba[1] << " " << state.ba[2];
//                         state_mutex_.unlock();
//                     } else {
//                         // 单次机械编排
//                         // Do a single INS mechanization
//                         MISC::insMechanization(integration_config_, imu_pre, imu_cur, state);
//                         ins_window_.back().second = state;
//                     }
//                 } else {
//                     // Only reserve certain INS in the window during initialization
//                     if (ins_window_.size() > MAXIMUM_INS_NUMBER) {
//                         ins_window_.pop_front();
//                     }
//                 }
//                 // 融合状态
//                 // Fusion process
//                 if (gvinsstate_ == GVINS_INITIALIZING) {
//                     {                    
//                         std::lock_guard<std::mutex> lock(frame_ready_mutex_);
//                         if (isframeready_ && state_mutex_.try_lock()) {
//                             // 初始化参数
//                             // GVINS initialization using GNSS/INS initialization
//                             if (gvinsInitialization()) {
//                                 gvinsstate_ = GVINS_INITIALIZING_VIO;
//                                 // 初始化时需要重新积分
//                                 // Redo INS mechanization
//                                 isoptimized_ = true;
//                             }
//                             isframeready_ = false;
//                             state_mutex_.unlock();
//                             continue;
//                         }
//                     }
//                     frame_ready_cv_.notify_one();
//                 } else if (gvinsstate_ == GVINS_INITIALIZING_VIO) {
//                     // 仅加入关键帧节点, 而不进行优化
//                     // Add new time node during the initialization of the visual system
//                     {                    
//                         std::lock_guard<std::mutex> lock(frame_ready_mutex_);
//                         if ((isframeready_) && state_mutex_.try_lock()) {
//                             if (isframeready_ && (keyframes_.front()->stamp() < ins_window_.back().first.time)) {
//                                 addNewKeyFrameTimeNode();
//                                 isframeready_ = false;
//                                 isvisualobs_  = true;
//                             }
//                             state_mutex_.unlock();
//                             if (isvisualobs_ && map_->isMaximumKeframes()) {
//                                 optimization_sem_.notify_one();
//                             }
//                         }
//                     }
//                     frame_ready_cv_.notify_one();
//                 } else if (gvinsstate_ >= GVINS_TRACKING_INITIALIZING) {
//                     if ((isframeready_) && state_mutex_.try_lock()) {
//                         if (isframeready_ && (keyframes_.front()->stamp() < ins_window_.back().first.time)) {
//                             addNewKeyFrameTimeNode();
//                             isframeready_ = false;
//                             isvisualobs_  = true;
//                         }
//                         state_mutex_.unlock();
//                         // Release the optimization semaphore
//                         if (isvisualobs_) {
//                             optimization_sem_.notify_one();
//                         }
//                     }
//                 }
//                 // 用于输出
//                 // For output only
//                 state = ins_window_.back().second;
//             } // INS
//             // 总是输出最新的INS机械编排结果, 不占用INS锁
//             // Always output the INS results
//             if (gvinsstate_ > GVINS_TRACKING_INITIALIZING) {//GVINS_TRACKING_INITIALIZING
//                 MISC::writeNavResult(integration_config_, state, imuerrfilesaver_, trajfilesaver_);
//             }
//         } // IMU BUFFER
//     }     // While
// }

void GVINS::runFusion() {
    IMU imu_pre, imu_cur;
    IntegrationState state;

    LOGI << "Fusion thread is started";

    while (!isfinished_) {
        // 等待信号（短时持锁仅用于等待）
        {
            std::unique_lock<std::mutex> lock(fusion_mutex_);
            fusion_sem_.wait(lock);
            if (isfinished_) break;
        }

        // TS("fusion_sem_ is lock");

        // 处理 IMU 缓存中的每个样本（一次处理一个样本，尽量缩短各锁的持有时间）
        while (true) {
            // 1) 迅速从 imu_buffer_ 中取出一个样本（只在 imu_buffer_mutex_ 下临界区短暂操作）
            {
                std::lock_guard<std::mutex> ib_lock(imu_buffer_mutex_);
                if (imu_buffer_.empty()) {
                    // TS("imu_buffer is empty");
                    break; // 缓存耗尽，退出 inner loop
                }
                imu_pre = imu_cur;
                imu_cur = imu_buffer_.front();
                imu_buffer_.pop();
            }

            // 2) 把新的 IMU 时间点压入 ins_window_（在 ins_mutex_ 下操作）并读取上一个 INS 状态快照
            {
                std::lock_guard<std::mutex> ins_lock(ins_mutex_);
                if (!ins_window_.empty()) state = ins_window_.back().second; // 读取上一时刻状态作为基础
                ins_window_.emplace_back(imu_cur, IntegrationState());
            }

            // 3) 如果处于已初始化后的阶段，执行机械编排/重积分逻辑
            if (gvinsstate_ > GVINS_INITIALIZING) {
                // 优先尝试拿到 state_mutex_ 去执行较重的 redoInsMechanization（如果有 isoptimized_ 标志）
                // TS("wait get isoptimized_");
                // bool opt_expected = true;
                // if (isoptimized_.compare_exchange_strong(opt_expected, false, std::memory_order_acq_rel)) {
                //     // try_lock 短暂抢占：能拿到就做重做积分，否则退化为单步积分
                //     TS("check state_mutex before redo");
                //     if (state_mutex_.try_lock()) {
                //         // 获取来自优化器的最新状态并重新机械编排
                //         auto start = std::chrono::high_resolution_clock::now();

                //         {
                //             std::lock_guard<std::mutex> ins_lock2(ins_mutex_);
                //             state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
                //             MISC::redoInsMechanization_new(integration_config_, state, reserved_ins_num_, ins_window_);
                //         }

                //         auto end = std::chrono::high_resolution_clock::now();
                //         double ms = std::chrono::duration<double, std::milli>(end - start).count();
                //         std::ostringstream oss;
                //         oss << std::fixed << std::setprecision(3) << ms;
                //         LOGI << "redoInsMechanization_new cost " << oss.str() << " ms";

                //         LOGI << "bias: " << state.bg[0] << " " << state.bg[1] << " " << state.bg[2]
                //              << " " << state.ba[0] << " " << state.ba[1] << " " << state.ba[2];
                //         state_mutex_.unlock();

                //         // 重做完成，通知等待的线程
                //         // ins_cv_.notify_all();
                //         TS("redo_done");
                //         frame_ready_cv_.notify_one();
                //     } else {
                //         // 无法拿到 state_mutex_ 时，先做单步机械编排以保证系统仍然前进
                //         MISC::insMechanization(integration_config_, imu_pre, imu_cur, state);
                //         std::lock_guard<std::mutex> ins_lock2(ins_mutex_);
                //         if (!ins_window_.empty()) ins_window_.back().second = state;
                //         LOGW << "InsMechanization once";
                //     }
                // } else {
                    // 普通单次机械编排
                    MISC::insMechanization(integration_config_, imu_pre, imu_cur, state);
                    std::lock_guard<std::mutex> ins_lock2(ins_mutex_);
                    if (!ins_window_.empty()) ins_window_.back().second = state;
                    // TS("isoptimized_ false, do single-step insMechanize");
                // }
            } else {
                // 初始化阶段：只保留窗口长度限制
                std::lock_guard<std::mutex> ins_lock2(ins_mutex_);
                if (ins_window_.size() > MAXIMUM_INS_NUMBER) ins_window_.pop_front();
            }

            if (gvinsstate_ == GVINS_INITIALIZING) {
                if (state_mutex_.try_lock()) {
                    bool should_process = isframeready_.load(std::memory_order_acquire);
                    if (should_process) {
                        if (gvinsInitialization()) {
                            gvinsstate_ = GVINS_INITIALIZING_VIO;
                            // 初始化完成后需要重新积分
                            isoptimized_.store(true, std::memory_order_release);
                        }
                        isframeready_.store(false, std::memory_order_release);
                    }
                    state_mutex_.unlock();
                    // 初始化成功后继续处理下一条 IMU（原代码用 continue 以避免落到输出），这里保持行为一致
                    if (should_process) {
                        frame_ready_cv_.notify_one();
                        continue;
                    }
                }
                frame_ready_cv_.notify_one();
            } else if (gvinsstate_ == GVINS_INITIALIZING_VIO) {
                if (state_mutex_.try_lock()) {
                    // 注意：keyframes_ 和 ins_window_ 的时间比较需要在持有 state_mutex_ 的情况下做（与原来行为一致）
                    bool should_process = isframeready_.load(std::memory_order_acquire) && 
                                          (!ins_window_.empty()) && 
                                          (keyframes_.front()->stamp() < ins_window_.back().first.time);
                    if (should_process) {
                        addNewKeyFrameTimeNode();
                        isframeready_.store(false, std::memory_order_release);
                        isvisualobs_.store(true, std::memory_order_release);
                        frame_ready_cv_.notify_one();
                    }
                    state_mutex_.unlock();
                    if (isvisualobs_.load(std::memory_order_acquire) && map_->isMaximumKeframes()) {
                        optimization_sem_.notify_one();
                    }
                }
            } else if (gvinsstate_ >= GVINS_TRACKING_INITIALIZING) {
                if (state_mutex_.try_lock()) {
                    bool need_notify_opt = false;
                    bool should_process = isframeready_.load(std::memory_order_acquire) && 
                                          (!ins_window_.empty()) && 
                                          (keyframes_.front()->stamp() < ins_window_.back().first.time);
                    if (should_process) {
                        addNewKeyFrameTimeNode();
                        isframeready_.store(false, std::memory_order_release);
                        // isvisualobs_.store(true, std::memory_order_release);
                        // frame_ready_cv_.notify_one();

                        // TS("shouldtrue");
                        bool expected = false;
                        if (isvisualobs_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                            // we changed false->true: do a single notify OUTSIDE锁 (下面解锁后再notify)
                            need_notify_opt = true;
                        }
                    }
                    state_mutex_.unlock();

                    if (need_notify_opt) {
                        optimization_sem_.notify_one();

                        // TS("opt_notify");
                    }
                }
            }

            // 5) 输出（将输出放在锁外以避免阻塞其他线程）
            {
                std::lock_guard<std::mutex> ins_lock3(ins_mutex_);
                if (!ins_window_.empty()) state = ins_window_.back().second;
            }

            if (gvinsstate_ > GVINS_TRACKING_INITIALIZING) {
                MISC::writeNavResult(integration_config_, state, imuerrfilesaver_, trajfilesaver_);
            }

        } // end while: process imu_buffer_
    } // end while: thread loop
}

void GVINS::runOptimization() {

    TimeCost timecost, timecost2;

    LOGI << "Optimization thread is started";
    while (!isfinished_) {
        Lock lock(optimization_mutex_);
        optimization_sem_.wait(lock);

        if (isvisualobs_.load(std::memory_order_acquire)) {
            // TS("opt_wakeup");

            timecost.restart();

            // 加锁, 保护状态量
            // Lock the state
            state_mutex_.lock();

            if (gvinsstate_ == GVINS_INITIALIZING_VIO) {
                Lock lock(tracking_mutex_);

                // GINS优化
                // GNSS/INS optimization
                bool isinitialized = gvinsInitializationOptimization();

                if (isinitialized) {
                    gvinsOptimization_old();
                    while (map_->isMaximumKeframes()) {
                        // 边缘化, 移除旧的观测, 按时间对齐到保留的最后一个关键帧
                        gvinsMarginalization();
                    }
                    parametersStatistic();

                    {
                        Lock lock3(ins_mutex_);
                        IntegrationState state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
                        // MISC::redoInsMechanization(integration_config_, state, reserved_ins_num_, ins_window_);
                        MISC::redoInsMechanization_new(integration_config_, state, reserved_ins_num_, ins_window_);
                    }

                    tracking_->setTrackChooseInitialing();
                    gvinsstate_ = GVINS_TRACKING_INITIALIZING;
                    LOGI << "GINS initialization is finished";
                } else {
                    slidewindow();
                    LOGW << "GINS initialization is not convergence";
                }
            } else if (gvinsstate_ >= GVINS_TRACKING_INITIALIZING) {

                if (map_->isMaximumKeframes()) {
                    gvinsstate_ = GVINS_TRACKING_NORMAL;
                }

                // 两次非线性优化并进行粗差剔除
                // Two-steps optimization with outlier culling
                // auto gon_start = std::chrono::high_resolution_clock::now();
                gvinsOptimization_new();
                // auto gon_end = std::chrono::high_resolution_clock::now();
                // double ms = std::chrono::duration<double, std::milli>(gon_end - gon_start).count();
                // std::ostringstream oss;
                // oss << std::fixed << std::setprecision(3) << ms;
                // LOGI << "gvinsOptimization_new cost " << oss.str() << " ms";

                timecost2.restart();

                // 移除所有窗口中间插入的非关键帧
                // Remove all non-keyframes time nodes
                gvinsRemoveAllSecondNewFrame();

                // 关键帧数量达到窗口大小, 需要边缘化操作, 并移除最老的关键帧及相关的GNSS和预积分观测, 由于计算力的问题,
                // 可能导致多个关键帧同时加入优化, 需要进行多次边缘化操作
                // Do marginalization
                while (map_->isMaximumKeframes()) {
                    // 边缘化, 移除旧的观测, 按时间对齐到保留的最后一个关键帧
                    gvinsMarginalization();
                }

                timecosts_[2] = timecost2.costInMillisecond();

                // 统计并输出视觉相关的参数
                // Log the statistic parameters
                parametersStatistic();
            }

            // 可视化
            // For visualization
            if (is_use_visualization_) {
                auto state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
                drawer_->updateMap(MISC::pose2Twc(MISC::stateToCameraPose(state, pose_b_c_)), state.time);
            }

            if (isvisualobs_.load(std::memory_order_acquire))
                isvisualobs_.store(false, std::memory_order_release);

            // Release the state lock
            state_mutex_.unlock();
            isoptimized_.store(true, std::memory_order_release);

            // LOGI << "Optimization costs " << timecost.costInMillisecond() << " ms with " << timecosts_[0] << " and "
            //      << timecosts_[1] << " with marginalization costs " << timecosts_[2];
            // TS("opt_done");
            // fusion_sem_.notify_one(); // 立即唤醒 fusion
            {            
                if (gvinsstate_ >= GVINS_TRACKING_INITIALIZING) {
                    // 1) 先在 state_mutex_ 保护下，把需要的“最新状态”拷贝出来
                    IntegrationState state_copy;
                    {
                        std::lock_guard<std::mutex> state_lk(state_mutex_);
                        state_copy = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
                    }

                    // 2) 再在 ins_mutex_ 保护下 redo INS window（只保护 INS 相关结构）
                    {
                        std::lock_guard<std::mutex> ins_lk(ins_mutex_);
                        MISC::redoInsMechanization_new(integration_config_, state_copy, reserved_ins_num_, ins_window_);
                    }

                    // 3) 唤醒等待 INS 的线程
                    ins_cv_.notify_all();
                }

                frame_ready_cv_.notify_all();
            }
        }
    }
}

// void GVINS::runTracking() {
//     Frame::Ptr frame;
//     Pose pose;
//     IntegrationState state, state0, state1;
//     std::deque<std::pair<IMU, IntegrationState>> ins_windows;
//     LOGI << "Tracking thread is started";
//     while (!isfinished_) {
//         Lock lock(tracking_mutex_);
//         tracking_sem_.wait(lock);
//         // 处理所有缓存
//         // Process all the frames
//         while (!frame_buffer_.empty()) {
//             TimeCost timecost;
//             Pose pose_b_c;
//             double td;
//             {
//                 Lock lock3(extrinsic_mutex_);
//                 pose_b_c = pose_b_c_;
//                 td       = td_b_c_;
//             }
//             // 获取初始位姿
//             // The prior pose from INS
//             if (gvinsstate_ >= GVINS_INITIALIZING_VIO){
//                 // 读取缓存
//                 {
//                     frame_buffer_mutex_.lock();
//                     frame = frame_buffer_.front();
//                     // 保证每个图像都有先验的惯导位姿
//                     // Wait until the INS is available
//                     ins_mutex_.lock();
//                     if (ins_window_.empty() || (ins_window_.back().second.time <= (frame->stamp() + td))) {
//                         ins_mutex_.unlock();
//                         frame_buffer_mutex_.unlock();
//                         // usleep(1000);
//                         std::this_thread::sleep_for(std::chrono::milliseconds(1));
//                         continue;
//                     }
//                     ins_mutex_.unlock();
//                     frame_buffer_.pop();
//                     frame_buffer_mutex_.unlock();
//                 }
//                 Lock lock2(ins_mutex_);
//                 frame->setStamp(frame->stamp() + td);
//                 frame->setTimeDelay(td);
//                 MISC::getCameraPoseFromInsWindow(ins_window_, pose_b_c, frame->stamp(), pose);
//                 frame->setPose(pose);
//             } else {
//                 // 读取缓存
//                 {
//                     frame_buffer_mutex_.lock();
//                     frame = frame_buffer_.front();
//                     frame_buffer_.pop();
//                     frame_buffer_mutex_.unlock();
//                 }
//                 Lock lock2(ins_mutex_);
//                 frame->setStamp(frame->stamp() + td);
//                 frame->setTimeDelay(td);
//                 // Pose pose0;
//                 // pose.R = Eigen::Matrix3d::Identity();
//                 // pose.t = Eigen::Vector3d::Zero();
//                 frame->setPose(pose_b_c);
//             }
//             TrackState trackstate = tracking_->track(frame);
//             if (trackstate == TRACK_LOST) {
//                 LOGE << "Tracking lost at " << Logging::doubleData(frame->stamp());
//             }
//             // while(isframeready_) {
//             //     usleep(1000);
//             // }
//             {
//                 std::unique_lock<std::mutex> lk(frame_ready_mutex_);
//                 // 如果 isframeready_ 为 true，等待直到被清除或超时（可调）
//                 if (isframeready_) {
//                     frame_ready_cv_.wait_for(lk, std::chrono::milliseconds(3), [this]{ return !isframeready_; });
//                 }
//             }
//             // 包括第一帧在内的所有关键帧, 跟踪失败时的当前帧也会成为新的关键帧
//             // All possible keyframes
//             if (tracking_->isNewKeyFrame() || (trackstate == TRACK_FIRST_FRAME) || trackstate == TRACK_LOST) {
//                 Lock lock3(keyframes_mutex_);
//                 keyframes_.push_back(frame);
//                 isframeready_ = true;
//                 LOGI << "Tracking cost " << timecost.costInMillisecond() << " ms";
//             } else if (trackstate == TRACK_CHECK) {
//                 {
//                     Lock lock3(keyframes_mutex_);
//                     keyframes_.push_back(frame);
//                     isframeready_ = true;
//                     LOGI << "Tracking cost " << timecost.costInMillisecond() << " ms";
//                     curframetime = frame->stamp();
//                     if (lastframetime == 0) lastframetime = curframetime;
//                 }
//                 {
//                     std::unique_lock<std::mutex> lk(frame_ready_mutex_);
//                     frame_ready_cv_.wait(lk, [this] { return !isframeready_; });
//                 }
//                 break;
//             }
//             // tracking_->gettracknum();
//         }
//     }
// }

void GVINS::runTracking() {
    Frame::Ptr frame;
    Pose pose;

    LOGI << "Tracking thread is started";
    while (!isfinished_) {
        // 等待唤醒
        {
            Lock lock(tracking_mutex_);
            tracking_sem_.wait(lock);
        }

        while (true) {
            TimeCost timecost;

            Pose pose_b_c;
            double td;
            {
                std::lock_guard<std::mutex> lk(extrinsic_mutex_);
                pose_b_c = pose_b_c_;
                td = td_b_c_;
            }

            // 尝试获取队首帧
            {
                std::lock_guard<std::mutex> fb_lk(frame_buffer_mutex_);
                if (frame_buffer_.empty()) break;
                frame = frame_buffer_.front();
            }

            double target_time = frame->stamp() + td;

            // VIO 初始化后，等待 INS 可用
            if (gvinsstate_ >= GVINS_INITIALIZING_VIO) {
                {
                    std::unique_lock<std::mutex> ins_lk(ins_mutex_);
                    ins_cv_.wait_for(ins_lk, std::chrono::milliseconds(10), [&]{
                        return isfinished_ || (!ins_window_.empty() && ins_window_.back().second.time > target_time);
                    });

                    if (isfinished_) break;

                    if (ins_window_.empty() || ins_window_.back().second.time <= target_time) {
                        continue; // INS 仍不可用，下一轮重试
                    }

                    MISC::getCameraPoseFromInsWindow(ins_window_, pose_b_c, target_time, pose);
                }

                // 安全地 pop 队首
                {
                    std::lock_guard<std::mutex> fb_lk(frame_buffer_mutex_);
                    if (!frame_buffer_.empty() && frame_buffer_.front() == frame)
                        frame_buffer_.pop();
                    else
                        continue;
                }

                frame->setStamp(target_time);
                frame->setTimeDelay(td);
                frame->setPose(pose);
            } else {
                // 初始化前直接使用 body->cam
                {
                    std::lock_guard<std::mutex> fb_lk(frame_buffer_mutex_);
                    if (!frame_buffer_.empty() && frame_buffer_.front() == frame)
                        frame_buffer_.pop();
                    else
                        continue;
                }

                frame->setStamp(target_time);
                frame->setTimeDelay(td);
                frame->setPose(pose_b_c);
            }

            // auto start = std::chrono::high_resolution_clock::now();

            TrackState trackstate = tracking_->track(frame);
            if (trackstate == TRACK_LOST) {
                LOGE << "Tracking lost at " << Logging::doubleData(frame->stamp());
            }

            // auto end = std::chrono::high_resolution_clock::now();
            // double ms = std::chrono::duration<double, std::milli>(end - start).count();
            // std::ostringstream oss;
            // oss << std::fixed << std::setprecision(3) << ms;
            // LOGI << "tracking cost 0: " << oss.str() << " ms";

            if (tracking_->isNewKeyFrame() || trackstate == TRACK_FIRST_FRAME || trackstate == TRACK_LOST) {
                {
                    std::lock_guard<std::mutex> lk(keyframes_mutex_);
                    keyframes_.push_back(frame);
                    isframeready_.store(true, std::memory_order_release);
                }

                // TS("T_track_push");

                {
                    std::unique_lock<std::mutex> lk(frame_ready_mutex_);
                    // auto wait_start = std::chrono::steady_clock::now();
                    
                    bool notified = frame_ready_cv_.wait_for(lk, 
                        std::chrono::milliseconds(100), 
                        [this]{ 
                            return !isframeready_.load(std::memory_order_acquire) || 
                                   isfinished_; 
                        });
                    
                    // auto wait_end = std::chrono::steady_clock::now();
                    // auto wait_ms = std::chrono::duration<double, std::milli>(
                    //     wait_end - wait_start).count();
                    
                    // if (!notified && !isfinished_) {
                    //     LOGW << "Keyframe wait timeout (" << wait_ms << "ms) at " 
                    //          << frame->stamp() << ", trackstate: " << trackstate;
                    // } else {
                    //     LOGW << "Keyframe consumed in " << wait_ms << "ms";
                    // }
                }

                // TS("T_track_wakeup");

                // LOGI << "Tracking cost " << timecost.costInMillisecond() << " ms";
            } else if (trackstate == TRACK_CHECK) {
                {
                    std::lock_guard<std::mutex> lk(keyframes_mutex_);
                    keyframes_.push_back(frame);
                    isframeready_.store(true, std::memory_order_release);
                    // LOGI << "Tracking cost " << timecost.costInMillisecond() << " ms";
                    curframetime = frame->stamp();
                    if (lastframetime == 0) lastframetime = curframetime;
                }

                {
                    std::unique_lock<std::mutex> lk(frame_ready_mutex_);
                    frame_ready_cv_.wait(lk, [this]{ return !isframeready_.load(std::memory_order_acquire) || isfinished_; });
                }

                break;
            }

            // tracking_->gettracknum();
        }
    }

    LOGI << "Tracking thread exits";
}


void GVINS::setFinished() {
    isfinished_ = true;

    // 释放信号量, 退出所有线程
    // Release all semaphores
    fusion_sem_.notify_all();
    tracking_sem_.notify_all();
    optimization_sem_.notify_all();

    tracking_thread_.join();
    optimization_thread_.join();
    fusion_thread_.join();

    if (is_use_visualization_) {
        drawer_->setFinished();
        drawer_thread_.join();
    }

    Quaterniond q_b_c = Rotation::matrix2quaternion(pose_b_c_.R);
    Vector3d t_b_c    = pose_b_c_.t;

    LOGW << "GVINS has finished processing";
    LOGW << "Estimated extrinsics: "
         << absl::StrFormat("(%0.6lf, %0.6lf, %0.6lf, %0.6lf), (%0.3lf, %0.3lf, "
                            "%0.3lf), %0.4lf",
                            q_b_c.x(), q_b_c.y(), q_b_c.z(), q_b_c.w(), t_b_c.x(), t_b_c.y(), t_b_c.z(), td_b_c_);

    Logging::shutdownLogging();
}

bool GVINS::gvinsInitialization() {

    // 缓存数据用于零速检测
    // Buffer for zero-velocity detection
    vector<IMU> imu_buff;
    for (const auto &ins : ins_window_) {
        auto &imu = ins.first;
        if ((imu.time > lastframetime) && (imu.time < curframetime)) {
            imu_buff.push_back(imu);
        }
    }
    if (imu_buff.size() < 40) {
        return false;
    }

    // 零速检测估计陀螺零偏和横滚俯仰角
    // Obtain the gyroscope biases and roll and pitch angles
    vector<double> average;
    static Vector3d bg{0, 0, 0};
    static Vector3d initatt{0, 0, 0};
    static bool is_has_zero_velocity = false;

    bool is_zero_velocity = MISC::detectZeroVelocity(imu_buff, imudatarate_, average);
    if (is_zero_velocity) {
        // 陀螺零偏
        bg = Vector3d(average[0], average[1], average[2]);
        bg *= imudatarate_;

        // 重力调平获取横滚俯仰角
        Vector3d fb(average[3], average[4], average[5]);
        fb *= imudatarate_;

        initatt[0] = -asin(fb[1] / integration_parameters_->gravity);
        initatt[1] = asin(fb[0] / integration_parameters_->gravity);

        LOGI << "Zero velocity get gyroscope bias " << bg.transpose() * 3600 * R2D << ", roll " << initatt[0] * R2D
             << ", pitch " << initatt[1] * R2D;
        is_has_zero_velocity = true;
    }

    // 从零速开始
    Vector3d velocity = Vector3d::Zero();
    Eigen::Quaterniond q(1, 0, 0, 0);
    Vector3d position = Vector3d::Zero();

    // 初始状态, 从上一秒开始
    // The initialization state
    auto state = IntegrationState{
        .time = lastframetime,
        .p    = position,
        .q    = q,
        .v    = velocity,
        .bg   = bg,
        .ba   = {0, 0, 0},
        .sodo = 0.0,
        .sg   = {0, 0, 0},
        .sa   = {0, 0, 0},
    };
    statedatalist_.emplace_back(Preintegration::stateToData(state, preintegration_options_));
    timelist_.push_back(lastframetime);
    // constructPrior(is_has_zero_velocity);//is_has_zero_velocity

    // 初始化重力和地球自转参数
    // The gravity and the Earth rotation rate
    integration_config_.gravity = Vector3d(0, 0, integration_parameters_->gravity);
    if (integration_config_.iswithearth) {
        integration_config_.iewn = Earth::iewn(integration_config_.origin, state.p);
    }

    // 计算第一秒的INS结果
    // Redo INS mechanization at the first second
    state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
    // MISC::redoInsMechanization(integration_config_, state, reserved_ins_num_, ins_window_);
    MISC::redoInsMechanization_new(integration_config_, state, reserved_ins_num_, ins_window_);

    LOGI << "Initialization at " << Logging::doubleData(curframetime);

    {
        Lock lock(keyframes_mutex_);
        Pose pose, pose_b_c;
        {
            Lock lock3(extrinsic_mutex_);
            pose_b_c = pose_b_c_;
        }
        for (const auto& keyframe : keyframes_) {
            MISC::getCameraPoseFromInsWindow(ins_window_, pose_b_c, keyframe->stamp(), pose);
            keyframe->setPose(pose);
        }

        auto frame = keyframes_.front();
        keyframes_.pop_front();
        map_->insertKeyFrame(frame);
    }

    addNewKeyFrameTimeNode();

    return true;
}

bool GVINS::gvinsInitializationOptimization() {
    //check imu observibility
    // 检查IMU的可观性
    // Step 11 ：保证IMU数据可用
    {
        Vector3d sum_g;
        for (const auto& preintegration: preintegrationlist_)
        {
            double dt = preintegration->deltaTime();
            Vector3d tmp_g = preintegration->delta_v() / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)preintegrationlist_.size() - 1);
        double var = 0;
        for (const auto& preintegration: preintegrationlist_)
        {
            double dt = preintegration->deltaTime();
            Vector3d tmp_g = preintegration->delta_v() / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)preintegrationlist_.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            // std::cout << "IMU excitation not enouth!";
            return false;
        }
    }

    std::map<int, double> int_frameid2_time_frameid;
    std::map<double, int> time_frameid2_int_frameid;

    int localwindow_id = 0;
    for (const auto& state: statedatalist_) {
        int_frameid2_time_frameid[localwindow_id] = state.time;
        time_frameid2_int_frameid[state.time] = localwindow_id;

        localwindow_id++;
    }

    {
        bool check_success = false;
        Eigen::Vector3d G = integration_config_.gravity;

        int num_imu_ = int_frameid2_time_frameid.size() - 1;

        // for (int i = 0; i < num_imu_; i++) {
        //     int j = i + 1;
        //     auto target1_tid = int_frameid2_time_frameid.at(j);
        //     const IntegrationBase& imu1 = all_image_frame.find(target1_tid)->second.getPreIntegration();
        //     CHECK((int_frameid2_time_frameid.at(j) - int_frameid2_time_frameid.at(i)) == imu1.sum_dt) <<
        //     int_frameid2_time_frameid.at(j) << " " << int_frameid2_time_frameid.at(i) << imu1.sum_dt;
        // }

        Eigen::Vector3d avgA;
        avgA.setZero();

        std::vector<int> is_bad(num_imu_, 0);
        for (int i = 0; i < num_imu_ - 1; i++) {
            const auto& delta_v = preintegrationlist_[i]->delta_v();
            auto sum_dt = preintegrationlist_[i]->deltaTime();
            Eigen::Vector3d acc = delta_v / sum_dt;
            if (std::abs(acc.norm() - G.norm()) / G.norm() < 5e-3)
                is_bad[i] = 1;
            avgA += delta_v / sum_dt;
        }

        int scoreSum = std::accumulate(is_bad.begin(), is_bad.end(), 0.0);

        avgA /= static_cast<double>(num_imu_);
        const double avgA_error = std::abs(avgA.norm() - G.norm()) / G.norm();

        if (avgA_error > 5e-3 and scoreSum <= 1)
            check_success = true;

        if (!check_success)
            return false;
    }

    Eigen::Map<Eigen::Matrix<double, 18, 1>> mix_vec(statedatalist_[0].mix);
    Eigen::Vector3d biasg = mix_vec.segment<3>(3);
    Eigen::Matrix3d pose_b_c_R = pose_b_c_.R;

    std::cout << "Start weight gyrobiasestimator!" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    if (!weightgyroBiasEstimator(int_frameid2_time_frameid, biasg)) {
        auto end = std::chrono::high_resolution_clock::now();
        LOGW << "Compute biasg cost " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms";
        return false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    LOGW << "Compute biasg cost " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms";

    Eigen::aligned_map<double, Eigen::Matrix3d> frame_rot;
    frame_rot.emplace(int_frameid2_time_frameid.at(0), Eigen::Matrix3d::Identity());

    Eigen::Matrix3d accumRot = Eigen::Matrix3d::Identity();
    for (size_t i = 1; i < int_frameid2_time_frameid.size(); i++) {
        std::memcpy(statedatalist_.at(i - 1).mix + 3, biasg.data(), 3 * sizeof(double));
        auto state = Preintegration::stateFromData(statedatalist_.at(i - 1), preintegration_options_);
        preintegrationlist_.at(i - 1)->reintegration(state);
        Eigen::Matrix3d dRcicj = pose_b_c_R.transpose() * preintegrationlist_.at(i - 1)->delta_q().toRotationMatrix() 
                                 * pose_b_c_R;
        // R_0_1 * R_1_2 * R_2_3....
        accumRot = accumRot * dRcicj;
        frame_rot.emplace(int_frameid2_time_frameid.at(i), accumRot);
    }
    std::memcpy(statedatalist_.at(int_frameid2_time_frameid.size() - 1).mix + 3, biasg.data(), 3 * sizeof(double));

    size_t num_view_ = int_frameid2_time_frameid.size();
    size_t num_pts_ = 0;

    std::vector<double> all_parallax;   // 每个点的最大旋转视差
    std::vector<double> all_weights;    // 对应点的权重

    int good_parallax_cnt = 0;
    {    
        // 计算仅旋转投影视差函数
        auto compute_max_parallax_mp = [&](const MapPoint::Ptr &mp) -> double {
            const auto &obs = mp->observations();
            double first_t = 0.0, last_t = 0.0;
            double prev_t = 0.0;
            Eigen::Vector3d prev_p = Eigen::Vector3d::Zero();
            bool have_prev = false;
            bool have_first = false;

            double sum_angles = 0.0;

            for (auto it = obs.begin(); it != obs.end(); ++it) {
                double t = it->first;
                if (frame_rot.count(t) == 0) continue;              // 只用我们有旋转的帧
                auto feat_wp = it->second;
                auto feat = feat_wp.lock();
                if (!feat) continue;

                const Eigen::Vector3d p = feat->point();           // normalized

                if (!have_first) {
                    first_t = t;
                    have_first = true;
                }

                if (have_prev) {
                    // 缓存 frame_rot 矩阵引用，避免多次查找
                    const Eigen::Matrix3d &R2 = frame_rot.at(t);
                    const Eigen::Matrix3d &R1 = frame_rot.at(prev_t);
                    Eigen::Matrix3d R21 = R2 * R1.transpose();

                    Eigen::Vector3d p_pred = R21 * prev_p;
                    p_pred.normalize();

                    Eigen::Vector3d crossv = p_pred.cross(p);
                    double sin_val = crossv.norm();
                    double cos_val = p_pred.dot(p);
                    double theta = std::atan2(sin_val, cos_val); // 更稳的角度计算

                    sum_angles += theta;
                }

                // 更新 prev
                prev_p = p;
                prev_t = t;
                have_prev = true;
                last_t = t;
            }

            if (!have_first) return 0.0;
            double time_span = last_t - first_t;
            if (time_span <= 1e-8) return 0.0;

            return sum_angles / time_span; // rad / sec
        };

        auto compute_max_parallax_sfm = [&](const drtSFMFeature &mp) -> double {
            const auto &obs = mp.obs;
            double first_t = 0.0, last_t = 0.0;
            double prev_t = 0.0;
            Eigen::Vector3d prev_p = Eigen::Vector3d::Zero();
            bool have_prev = false;
            bool have_first = false;

            double sum_angles = 0.0;

            for (auto it = obs.begin(); it != obs.end(); ++it) {
                double t = it->first;
                if (frame_rot.count(t) == 0) continue;
                const auto &feat = it->second;        // value type, 直接使用
                const Eigen::Vector3d p = feat.point(); // normalized

                if (!have_first) {
                    first_t = t;
                    have_first = true;
                }

                if (have_prev) {
                    const Eigen::Matrix3d &R2 = frame_rot.at(t);
                    const Eigen::Matrix3d &R1 = frame_rot.at(prev_t);
                    Eigen::Matrix3d R21 = R2 * R1.transpose();

                    Eigen::Vector3d p_pred = R21 * prev_p;
                    p_pred.normalize();

                    Eigen::Vector3d crossv = p_pred.cross(p);
                    double sin_val = crossv.norm();
                    double cos_val = p_pred.dot(p);
                    double theta = std::atan2(sin_val, cos_val);

                    sum_angles += theta;
                }

                prev_p = p;
                prev_t = t;
                have_prev = true;
                last_t = t;
            }

            if (!have_first) return 0.0;
            double time_span = last_t - first_t;
            if (time_span <= 1e-8) return 0.0;

            return sum_angles / time_span; // rad / sec
        };

        {
            auto [lock, sfm] = map_->SFMConstruct();
            for (const auto &kv : sfm) {
                const auto &sfm_feat = kv.second;

                if (sfm_feat.obs.size() < 3)
                    continue;

                double parallax = compute_max_parallax_sfm(sfm_feat);
                all_parallax.push_back(parallax);
                ++num_pts_;

                if (parallax > min_parallax_spd_)
                    good_parallax_cnt++;
            }
            
            // ------------ 处理 MapPoints --------------
            for (const auto &landmark : map_->landmarks()) {
                const auto &mp = landmark.second;
                const auto &obs = mp->observations();

                size_t count = 0;
                for (const auto &kv : obs)
                    if (std::find(timelist_.begin(), timelist_.end(), kv.first) != timelist_.end())
                        ++count;

                if (count < 3)
                    continue;

                double parallax = compute_max_parallax_mp(mp);
                all_parallax.push_back(parallax);
                ++num_pts_;

                if (parallax > min_parallax_spd_)
                    good_parallax_cnt++;
            }
        }

        LOGW << "good_parallax_cnt: " << good_parallax_cnt << " ;";

        // ---- 可观性判定 ----
        if (good_parallax_cnt < max_good_parallax_count_) {
            resetGyroBias(int_frameid2_time_frameid.size());
            return false;
        }

        {
            // 计算累积旋转
            Eigen::Matrix3d R_total = frame_rot.at(int_frameid2_time_frameid.at(int_frameid2_time_frameid.size() - 1));
            Eigen::AngleAxisd aa_total(R_total);
            double total_rotation_rad = aa_total.angle();
            
            LOGW << "Total rotation: " << total_rotation_rad * 57.3 << " deg, "
                << "good_parallax_cnt: " << good_parallax_cnt;
            
            // 旋转阈值
            if (total_rotation_rad < 0.05) {  // 15度
                LOGW << "Insufficient rotation for initialization";
                resetGyroBias(int_frameid2_time_frameid.size());
                return false;
            }
            
            // 可选：检查旋转分布（避免单轴旋转）
            Eigen::Vector3d rot_axis = aa_total.axis();
            if (std::abs(rot_axis.z()) > 0.95) {  // 纯yaw旋转
                LOGW << "Rotation too concentrated in yaw axis";
                // 可考虑放宽或继续，取决于你的需求
            }
        }

        if (!all_parallax.empty()) {
            // 找最大值
            double max_para = *std::max_element(all_parallax.begin(), all_parallax.end());

            // 防止除零
            if (max_para < 1e-8)
                max_para = 1e-8;

            all_weights.reserve(all_parallax.size());

            // --- 计算权重（线性归一化） ---
            for (double p : all_parallax) {
                double w = p / max_para;     // ∈[0,1]
                all_weights.push_back(w);
            }
        }
    }

    Eigen::MatrixXd LTL = Eigen::MatrixXd::Zero(num_view_ * 3 - 3, num_view_ * 3 - 3);
    Eigen::MatrixXd A_lr = Eigen::MatrixXd::Zero(num_pts_, 3 * num_view_);
    build_LTL(frame_rot, int_frameid2_time_frameid, time_frameid2_int_frameid, LTL, A_lr, all_weights);
    Eigen::VectorXd evectors = Eigen::VectorXd::Zero(3 * num_view_);

    if (!solve_LTL(LTL, evectors)) {
        resetGyroBias(int_frameid2_time_frameid.size());
        return false;
    }

    identify_sign(A_lr, evectors);

    // std::cout << "LTL solve evectors: " << evectors.transpose() << std::endl;

    std::vector<Eigen::Vector3d> velocity(num_view_);
    std::vector<Eigen::Vector3d> position(num_view_);
    std::vector<Eigen::Matrix3d> rotation(num_view_);
    for (size_t i = 0; i < num_view_; i++) {
        rotation[i] = frame_rot.at(int_frameid2_time_frameid.at(i)) * pose_b_c_R.transpose();
        position[i] = evectors.middleRows<3>(3 * i);
    }

    if (linearAlignment(int_frameid2_time_frameid, time_frameid2_int_frameid, velocity, position, rotation)) {
        return true;
    } else {
        resetGyroBias(int_frameid2_time_frameid.size());
        printf("solve g failed!\n");
        return false;
    }
}

void GVINS::resetGyroBias(const double N)
{
    Eigen::Vector3d biasg00 = Eigen::Vector3d::Zero();
    for (int i = 1; i < N; i++) {
        std::memcpy(statedatalist_.at(i - 1).mix + 3, biasg00.data(), 3 * sizeof(double));
        auto state = Preintegration::stateFromData(statedatalist_.at(i - 1), preintegration_options_);
        preintegrationlist_.at(i - 1)->reintegration(state);
    }
    std::memcpy(statedatalist_.at(N - 1).mix + 3, biasg00.data(), 3 * sizeof(double));
}

bool GVINS::weightgyroBiasEstimator(const std::map<int, double> &int_frameid2_time_frameid, Eigen::Vector3d &biasg)
{
    std::vector<std::vector<double>> ws;
    solve_ws(int_frameid2_time_frameid, ws);

    Eigen::IOFormat LogFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1e-5);
    // std::cout << "before bg: " << biasg.transpose() << std::endl;
    LOGW << "before bg: " << biasg.format(LogFmt);

    int num_obs = 0;
    std::vector<Eigen::Vector3d> fis;
    std::vector<Eigen::Vector3d> fjs;

    std::vector<Eigen::Vector2d> fis_img;
    std::vector<Eigen::Vector2d> fjs_img;
    Eigen::Matrix3d pose_b_c_R = pose_b_c_.R;

    for (size_t i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
        auto target1_tid = int_frameid2_time_frameid.at(i);
        auto target2_tid = int_frameid2_time_frameid.at(i + 1);

        fis.clear();
        fjs.clear();
        fis_img.clear();
        fjs_img.clear();

        {
            auto [lock, sfm] = map_->SFMConstruct();
            for (const auto &pts: sfm) {
                // if(pts.second.obs.size() < 3) continue;
                if (pts.second.obs.find(target1_tid) != pts.second.obs.end() &&
                    pts.second.obs.find(target2_tid) != pts.second.obs.end()) {
                    auto &feature1 = pts.second.obs.at(target1_tid);
                    auto &feature2 = pts.second.obs.at(target2_tid);

                    ++num_obs;

                    fis.push_back(feature1.point());
                    fjs.push_back(feature2.point());

                    fis_img.push_back(feature1.keyPoint());
                    fjs_img.push_back(feature2.keyPoint());
                }
            }
        }

        {
            for (auto &landmark: map_->landmarks()) {
                auto obs = landmark.second->observations();
                // if(obs.size() < 3) continue;
                if (obs.find(target1_tid) != obs.end() &&
                    obs.find(target2_tid) != obs.end()) {
                    auto feature1 = obs.at(target1_tid).lock();
                    auto feature2 = obs.at(target2_tid).lock();

                    ++num_obs;

                    fis.push_back(feature1->point());
                    fjs.push_back(feature2->point());

                    fis_img.push_back(feature1->uv());
                    fjs_img.push_back(feature2->uv());
                }
            }
        }

        if (fis.size() == 0 || fjs.size() == 0) continue;

        //自动求导
        ceres::CostFunction *eigensolver_cost_function = WeightBiasSolverCostFunctor::Create(fis, fjs, ws[i], 
                                                                                        Eigen::Quaterniond(pose_b_c_R),
                                                                                        preintegrationlist_[i]);
        problem.AddResidualBlock(eigensolver_cost_function, loss_function, biasg.data());

    } // end frame to frame loop

    std::cout << "number observation: " << num_obs << std::endl;

    double avg_observation = num_obs / int_frameid2_time_frameid.size();

    if(avg_observation < 30) {
        std::cout << "invalid number observation: " << avg_observation << std::endl;
        // throw -1;
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    // options.min_linear_solver_iterations = 10;
    options.gradient_tolerance = 1e-20;
    options.function_tolerance = 1e-20;
    options.parameter_tolerance = 1e-20;
    // options.jacobi_scaling = false;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;

    try
    {
        ceres::Solve(options, &problem, &summary);
    }
    catch(...)
    {
        return false;
    }

    if (summary.termination_type != ceres::TerminationType::CONVERGENCE) {
        std::cout << "not converge" << std::endl;
        return false;
    }
    // std::cout << std::fixed << std::setprecision(10);
    // std::cout << "after bg: " << biasg.transpose() << std::endl;
    LOGW << "after bg: " << biasg.format(LogFmt);

    return true;

}

void GVINS::solve_ws(const std::map<int, double> &int_frameid2_time_frameid, std::vector<std::vector<double>> &ws) {
    #pragma region 循环计算每两帧之间特征点权重
    // size_t max_iterations = 50;
    // double noise_bound_sq = 0.05;
    // double mu = 1;
    // double cost_ = 0;
    // double sum_w = 0;
    // double prev_cost = 0;
    // double gnc_factor = 1.3;
    // double cost_threshold = 1e-12;
    // bool it_success = false;
    // size_t num_obs = 0;

    // std::vector<Eigen::Vector3d> fis;
    // std::vector<Eigen::Vector3d> fjs;

    // std::vector<Eigen::Vector2d> fis_img;
    // std::vector<Eigen::Vector2d> fjs_img;

    // for (size_t i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
    //     auto target1_tid = int_frameid2_time_frameid.at(i);
    //     auto target2_tid = int_frameid2_time_frameid.at(i + 1);

    //     num_obs = 0;
    //     fis.clear();
    //     fjs.clear();
    //     fis_img.clear();
    //     fjs_img.clear();

    //     {
    //         auto [lock, sfm] = map_->SFMConstruct();
    //         for (const auto &pts: sfm) {
    //             // if(pts.second.obs.size() < 3) continue;
    //             if (pts.second.obs.find(target1_tid) != pts.second.obs.end() &&
    //                 pts.second.obs.find(target2_tid) != pts.second.obs.end()) {
    //                 auto &feature1 = pts.second.obs.at(target1_tid);
    //                 auto &feature2 = pts.second.obs.at(target2_tid);

    //                 ++num_obs;

    //                 fis.push_back(feature1.point());
    //                 fjs.push_back(feature2.point());

    //                 fis_img.push_back(feature1.keyPoint());
    //                 fjs_img.push_back(feature2.keyPoint());
    //             }
    //         }
    //     }

    //     {
    //         for (auto &landmark: map_->landmarks()) {
    //             auto obs = landmark.second->observations();
    //             // if(obs.size() < 3) continue;
    //             if (obs.find(target1_tid) != obs.end() &&
    //                 obs.find(target2_tid) != obs.end()) {
    //                 auto feature1 = obs.at(target1_tid).lock();
    //                 auto feature2 = obs.at(target2_tid).lock();

    //                 ++num_obs;

    //                 fis.push_back(feature1->point());
    //                 fjs.push_back(feature2->point());

    //                 fis_img.push_back(feature1->uv());
    //                 fjs_img.push_back(feature2->uv());
    //             }
    //         }
    //     }

    //    if (fis.size() == 0 || fjs.size() == 0) {
    //        std::vector<double> w(1, -1.0);
    //        ws.push_back(w);
    //        continue;
    //    }

    //    std::vector<double> w(fis.size(), 1.0);
    //    // no drt_init_weight
    //    // ws.push_back(w);

    //     Eigen::VectorXd res(fis.size());
    //     std::shared_ptr<PreintegrationBase> imu1 = preintegrationlist_[i];

    //     for (size_t i = 0; i < max_iterations; ++i) {
    //         solve_res(fis, fjs, w, imu1, res);

    //         if (i == 0) {
    //             double max_residual = res.maxCoeff();
    //             mu = 1 / (2 * max_residual / noise_bound_sq - 1);
    //             if (mu <= 0) {
    //                 it_success = true;
    //                 break;
    //             }
    //         }

    //         double th1 = (mu + 1) / mu * noise_bound_sq;
    //         double th2 = mu / (mu + 1) * noise_bound_sq;
    //         cost_ = 0;
    //         sum_w = 0;
    //         for (size_t j = 0; j < fis.size(); ++j) {
    //             cost_ += w[j] * res(j);
    //             sum_w += w[j];
    //             if (res(j) >= th1) {
    //                 w[j] = 0;
    //             } else if (res(j) <= th2) {
    //                 w[j] = 1;
    //             } else {
    //                 w[j] = sqrt(noise_bound_sq * mu * (mu + 1) / res(j)) - mu;
    //                 assert(w[j] >= 0 && w[j] <= 1);
    //             }
    //         }

    //         double cost_diff = std::abs(cost_ - prev_cost);

    //         appendLog(i, sum_w, cost_, num_obs, mu, "/sad/catkin_ws/ex_logs", "biasg_changes.txt");

    //         if (cost_diff < cost_threshold) {
    //             it_success = sum_w > 10;
    //             break;
    //         }

    //         // Increase mu
    //         mu = mu * gnc_factor;
    //         prev_cost = cost_;
    //     }

    //     if (it_success) {
    //         ws.push_back(w);
    //     } else {
    //         std::fill(w.begin(), w.end(), 1.0);
    //         ws.push_back(w);
    //     }

    // } 
    #pragma endregion

    const int frame_pair_num = int_frameid2_time_frameid.size() - 1;
    ws.resize(frame_pair_num);

    parallel_for(blocked_range<int>(0, frame_pair_num),
                    [&](const blocked_range<int> &r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            const double target1_tid = int_frameid2_time_frameid.at(i);
            const double target2_tid = int_frameid2_time_frameid.at(i + 1);

            size_t num_obs = 0;
            std::vector<Eigen::Vector3d> fis, fjs;
            std::vector<Eigen::Vector2d> fis_img, fjs_img;

            // 收集特征点（线程安全：读取）
            {
                auto [lock, sfm] = map_->SFMConstruct();
                for (const auto &pts : sfm) {
                    // if(pts.second.obs.size() < 3) continue;
                    if (pts.second.obs.find(target1_tid) != pts.second.obs.end() &&
                        pts.second.obs.find(target2_tid) != pts.second.obs.end()) {
                        auto &feature1 = pts.second.obs.at(target1_tid);
                        auto &feature2 = pts.second.obs.at(target2_tid);

                        ++num_obs;

                        fis.push_back(feature1.point());
                        fjs.push_back(feature2.point());

                        fis_img.push_back(feature1.keyPoint());
                        fjs_img.push_back(feature2.keyPoint());
                    }
                }
            }
            {
                for (auto &landmark : map_->landmarks()) {
                    auto obs = landmark.second->observations();
                    // if (obs.size() < 3) continue;
                    if (obs.find(target1_tid) != obs.end() &&
                        obs.find(target2_tid) != obs.end()) {
                        auto feature1 = obs.at(target1_tid).lock();
                        auto feature2 = obs.at(target2_tid).lock();

                        ++num_obs;

                        fis.push_back(feature1->point());
                        fjs.push_back(feature2->point());

                        fis_img.push_back(feature1->uv());
                        fjs_img.push_back(feature2->uv());
                    }
                }
            }

            std::vector<double> w(fis.size(), 1.0);
            Eigen::VectorXd res(fis.size());
            bool it_success = false;

            if (!fis.empty()) {
                auto imu1 = preintegrationlist_[i];
                double mu = 1, cost_ = 0, prev_cost = 0, sum_w = 0;
                const double noise_bound_sq = noise_bound_sq_;//0.18
                const double gnc_factor = 1.4;
                const double cost_threshold = 1e-12;

                for (int iter = 0; iter < 50; ++iter) {
                    solve_res(fis, fjs, w, imu1, res);
                    if (iter == 0) {
                        double max_res = res.maxCoeff();
                        mu = 1.0 / (2.0 * max_res / noise_bound_sq - 1);
                        if (mu <= 0) {
                            it_success = true;
                            break;
                        }
                    }

                    double th1 = (mu + 1) / mu * noise_bound_sq;
                    double th2 = mu / (mu + 1) * noise_bound_sq;
                    cost_ = sum_w = 0;

                    for (int j = 0; j < res.size(); ++j) {
                        cost_ += w[j] * res[j];
                        sum_w += w[j];
                        if (res[j] >= th1)           w[j] = 0;
                        else if (res[j] <= th2)      w[j] = 1;
                        else {
                            w[j] = std::sqrt(noise_bound_sq * mu * (mu + 1) / res[j]) - mu;
                            assert(w[j] >= 0 && w[j] <= 1);
                        }
                    }

                    appendLog(iter, sum_w, cost_, num_obs, mu, "/sad/catkin_ws/ex_logs", "biasg_changes.txt");

                    if (std::abs(cost_ - prev_cost) < cost_threshold) {
                        it_success = sum_w > 15;
                        break;
                    }
                    mu *= gnc_factor;
                    prev_cost = cost_;
                }
            }

            if (!it_success) {
                std::fill(w.begin(), w.end(), 1.0);
            }

            ws[i] = std::move(w);
        }
    });

}

void GVINS::solve_res(const std::vector<Eigen::Vector3d> &fis, const std::vector<Eigen::Vector3d> &fjs, const std::vector<double> &w,
                        std::shared_ptr<PreintegrationBase> imu1, Eigen::VectorXd &res) {
    Eigen::Map<Eigen::Matrix<double, 18, 1>> mix_vec(statedatalist_[0].mix);
    Eigen::Vector3d biasg = mix_vec.segment<3>(3);
    // Eigen::Vector3d biasg = Eigen::Vector3d::Zero();
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1e-5);
    Eigen::Matrix3d pose_b_c_R = pose_b_c_.R;

    //自动求导
    ceres::CostFunction *eigensolver_cost_function = WeightBiasSolverCostFunctor::Create(fis, fjs, w,
                                                                                    Eigen::Quaterniond(pose_b_c_R),
                                                                                    imu1);
    problem.AddResidualBlock(eigensolver_cost_function, loss_function, biasg.data());

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    // options.min_linear_solver_iterations = 10;
    options.gradient_tolerance = 1e-20;
    options.function_tolerance = 1e-20;
    options.parameter_tolerance = 1e-20;
    // options.jacobi_scaling = false;
    options.linear_solver_type = ceres::DENSE_SCHUR;//ceres::DENSE_QR
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    size_t nks_num = fis.size();
    Eigen::MatrixXd nks(nks_num, 3);
    Eigen::MatrixXd nks_w(nks_num, 3);
    Eigen::Matrix3d jacobina_q_bg = imu1->dq_dbg();  // ∂q/∂bg
    Eigen::Quaterniond _qic = Eigen::Quaterniond(pose_b_c_R);  // IMU到相机的旋转
    Eigen::Quaterniond qjk_imu = imu1->delta_q();  // IMU之间的预积分旋转

    // 使用优化得到的biasg修正旋转
    Eigen::Vector3d deltaBg = biasg - imu1->bg();
    Eigen::Matrix<double, 3, 1> jacobian_bg = jacobina_q_bg.cast<double>() * deltaBg;
    
    Eigen::Matrix<double, 4, 1> qij_tmp;
    ceres::AngleAxisToQuaternion(jacobian_bg.data(), qij_tmp.data());
    Eigen::Quaternion<double> qij(qij_tmp(0), qij_tmp(1), qij_tmp(2), qij_tmp(3));
    qjk_imu = qjk_imu * qij;  // 修正之后的IMU旋转

    Eigen::Quaterniond qcjk = _qic.inverse() * qjk_imu * _qic;  // IMU转到相机坐标系的变换
    for (size_t i = 0; i < nks_num; i++) {
        Eigen::Vector3d f1 = fis[i].normalized();  // f1 转到同一坐标系
        Eigen::Vector3d f2 = qcjk.inverse() * fjs[i].normalized();  // f2 也转到同一坐标系
        Eigen::Vector3d n_raw = cross_product_matrix(f1) * f2;
        Eigen::Vector3d nk = n_raw.normalized();
        nks.row(i) = n_raw.transpose();  // 计算f1, f2构成的平面的法向量
        nks_w.row(i) = w[i] * n_raw.transpose();  // 计算f1, f2构成的平面的法向量
    }

    // ========================= Solve Problem by Eigen's SVD =======================
    // std::cout << "Solve Problem by Eigen's SVD" << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(nks_w, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //  if (svd.info() != Eigen::Success)
    //    OPENMVG_LOG_ERROR << "SVD solver failure - expect to have invalid output";
    Eigen::Vector3d t = svd.matrixV().col(2).normalized();
    // 每个面法线和t之间的余弦平方
    res.resize(nks_num);
    for (size_t i = 0; i < nks_num; ++i) {
        Eigen::Vector3d f1 = fis[i].normalized();
        Eigen::Vector3d f2 = qcjk.inverse() * fjs[i].normalized();
        Eigen::Vector3d n_raw = cross_product_matrix(f1) * f2;
        Eigen::Vector3d nk = n_raw.normalized();
        double val = n_raw.dot(t);
        res[i] = (val * val);//  / (n_raw.squaredNorm())
    }

    // Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
    // std::vector<int> valid_idx;
    // for (size_t i = 0; i < nks_num; ++i) {
    //     Eigen::Vector3d f1 = fis[i].normalized();
    //     Eigen::Vector3d f2 = qcjk.inverse() * fjs[i].normalized();
    //     Eigen::Vector3d n_raw = f1.cross(f2);
    //     double n2 = n_raw.squaredNorm();
    //     if (n2 < 1e-12) continue; // skip nearly collinear
    //     M += w[i] * (n_raw * n_raw.transpose()); // weight includes angle magnitude
    //     valid_idx.push_back(i);
    // }
    // // solve min eigenvector
    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
    // Eigen::Vector3d t = es.eigenvectors().col(0); // smallest eigenval
    // t.normalize();

    // // compute per-sample residuals (weighted or normalized as you prefer)
    // res.resize(nks_num);
    // res.setZero();
    // for (int idx : valid_idx) {
    //     Eigen::Vector3d f1 = fis[idx].normalized();
    //     Eigen::Vector3d f2 = qcjk.inverse() * fjs[idx].normalized();
    //     Eigen::Vector3d n_raw = f1.cross(f2);
    //     double val = (n_raw.dot(t));
    //     // e.g., normalized residual:
    //     res[idx] = (val*val) / (n_raw.squaredNorm() + 1e-12);
    // }
}

void GVINS::build_LTL(const Eigen::aligned_map<double, Eigen::Matrix3d> &frame_rot, const std::map<int, double> &int_frameid2_time_frameid, 
                      const std::map<double, int> &time_frameid2_int_frameid, Eigen::MatrixXd &LTL, Eigen::MatrixXd &A_lr, std::vector<double> &all_weights) {
    int num_view_ = int_frameid2_time_frameid.size();
    int track_id = 0;
    {
        auto [lock, sfm] = map_->SFMConstruct();
        for (const auto &pt: sfm) {
            const auto &obs = pt.second.obs;
            if (obs.size() < 3) continue;
            double lbase_view_id = 0;
            double rbase_view_id = 0;
            select_base_views(frame_rot,
                              obs, 
                              int_frameid2_time_frameid, 
                              time_frameid2_int_frameid, 
                              lbase_view_id,
                              rbase_view_id);
            if (lbase_view_id == 0 ||rbase_view_id == 0) continue;
            //原本中的L矩阵，[B, C, D, ...]
            Eigen::MatrixXd tmp_LiGT_vec = Eigen::MatrixXd::Zero(3, num_view_ * 3);
            // [Step.3 in Pose-only algorithm]: calculate local L matrix,
            for (const auto &frame: obs) {
                // the current view id
                double i_view_id = frame.first;
                if (time_frameid2_int_frameid.find(i_view_id) == time_frameid2_int_frameid.end()) continue;
                //使用X_i和X_l，可以计算t_il，引入r view计算l view的深度，对应公式(11)
                if (i_view_id != lbase_view_id) {
                    //公式(21)
                    Eigen::Matrix3d xi_cross = cross_product_matrix(frame.second.point());
                    Eigen::Matrix3d R_cicl =
                            frame_rot.at(i_view_id).transpose() * frame_rot.at(lbase_view_id);
                    Eigen::Matrix3d R_crcl =
                            frame_rot.at(rbase_view_id).transpose() * frame_rot.at(lbase_view_id);
                    Eigen::Vector3d a_lr_tmp_t =
                            cross_product_matrix(R_crcl * obs.at(lbase_view_id).point()) *
                            obs.at(rbase_view_id).point();
                    Eigen::RowVector3d a_lr_t =
                            a_lr_tmp_t.transpose() * cross_product_matrix(obs.at(rbase_view_id).point());
                    // combine all a_lr vectors into a matrix form A, i.e., At > 0
                    // a_lr * trl = 0 -> alr * Rrw *(twl - twr) = 0 公式写trl对应原文tlr, 为了把相对的t转为全局的，所以乘了个R
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) =
                            a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) =
                            -a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    // theta_lr
                    Eigen::Vector3d theta_lr_vector = cross_product_matrix(obs.at(rbase_view_id).point())
                                                        * R_crcl
                                                        * obs.at(lbase_view_id).point();
                    double theta_lr = theta_lr_vector.squaredNorm();
                    // calculate matrix B [rbase_view_id]
                    // 对应公(18) 也能看出来变量里把transpose省略掉了，B对应right view的全局t
                    Eigen::Matrix3d Coefficient_B =
                            xi_cross * R_cicl * obs.at(lbase_view_id).point() * a_lr_t *
                            frame_rot.at(rbase_view_id).transpose();
                    // calculate matrix C [i_view_id]
                    Eigen::Matrix3d Coefficient_C = theta_lr * cross_product_matrix(obs.at(i_view_id).point()) *
                                                    frame_rot.at(i_view_id).transpose();
                    // calculate matrix D [lbase_view_id]
                    Eigen::Matrix3d Coefficient_D = -(Coefficient_B + Coefficient_C);
                    // calculate temp matrix L for a single 3D matrix
                    tmp_LiGT_vec.setZero();
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) += Coefficient_B;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(i_view_id) * 3) += Coefficient_C;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) += Coefficient_D;
                    // calculate LtL submodule
                    Eigen::MatrixXd LTL_l_row = Coefficient_D.transpose() * tmp_LiGT_vec * all_weights[track_id];
                    Eigen::MatrixXd LTL_r_row = Coefficient_B.transpose() * tmp_LiGT_vec * all_weights[track_id];
                    Eigen::MatrixXd LTL_i_row = Coefficient_C.transpose() * tmp_LiGT_vec * all_weights[track_id];
                    // assignment for LtL (except for the reference view id)
                    // #pragma omp critical
                    {
                        if (time_frameid2_int_frameid.at(lbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(lbase_view_id) * 3 - 3) += LTL_l_row.rightCols(
                                    LTL_l_row.cols() - 3);
                        if (time_frameid2_int_frameid.at(rbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(rbase_view_id) * 3 - 3) += LTL_r_row.rightCols(
                                    LTL_r_row.cols() - 3);
                        if (time_frameid2_int_frameid.at(i_view_id) > 0)
                            LTL.middleRows<3>(time_frameid2_int_frameid.at(i_view_id) * 3 - 3) += LTL_i_row.rightCols(
                                    LTL_i_row.cols() - 3);
                    }
                }
            }
            ++track_id;
        }
    }
    {
        for (const auto &landmark: map_->landmarks()) {
            const auto &obs = landmark.second->observations();
            size_t count = 0;
            for (const auto &kv : obs) {
                if (std::find(timelist_.begin(), timelist_.end(), kv.first) != timelist_.end()) {
                    ++count;
                }
            }
            if (count < 3) continue;
            double lbase_view_id = 0;
            double rbase_view_id = 0;
            select_base_views(frame_rot,
                              obs, 
                              int_frameid2_time_frameid, 
                              time_frameid2_int_frameid, 
                              lbase_view_id,
                              rbase_view_id);
            if (lbase_view_id == 0 ||rbase_view_id == 0) continue;
            //原本中的L矩阵，[B, C, D, ...]
            Eigen::MatrixXd tmp_LiGT_vec = Eigen::MatrixXd::Zero(3, num_view_ * 3);
            // [Step.3 in Pose-only algorithm]: calculate local L matrix,
            for (const auto &frame: obs) {
                // the current view id
                double i_view_id = frame.first;
                if (time_frameid2_int_frameid.find(i_view_id) == time_frameid2_int_frameid.end()) continue;
                //使用X_i和X_l，可以计算t_il，引入r view计算l view的深度，对应公式(11)
                if (i_view_id != lbase_view_id) {
                    auto xi = frame.second.lock();
                    auto obs_l = obs.at(lbase_view_id).lock();
                    auto obs_r = obs.at(rbase_view_id).lock();
                    auto obs_i = obs.at(i_view_id).lock();
                    if (!(xi && obs_l && obs_r)) continue;

                    //公式(21)
                    Eigen::Matrix3d xi_cross = cross_product_matrix(xi->point());
                    Eigen::Matrix3d R_cicl =
                            frame_rot.at(i_view_id).transpose() * frame_rot.at(lbase_view_id);
                    Eigen::Matrix3d R_crcl =
                            frame_rot.at(rbase_view_id).transpose() * frame_rot.at(lbase_view_id);
                    Eigen::Vector3d a_lr_tmp_t =
                            cross_product_matrix(R_crcl * obs_l->point()) *
                            obs_r->point();
                    Eigen::RowVector3d a_lr_t =
                            a_lr_tmp_t.transpose() * cross_product_matrix(obs_r->point());
                    // combine all a_lr vectors into a matrix form A, i.e., At > 0
                    // a_lr * trl = 0 -> alr * Rrw *(twl - twr) = 0 公式写trl对应原文tlr, 为了把相对的t转为全局的，所以乘了个R
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) =
                            a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    A_lr.row(track_id).block<1, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) =
                            -a_lr_t * frame_rot.at(rbase_view_id).transpose();
                    // theta_lr
                    Eigen::Vector3d theta_lr_vector = cross_product_matrix(obs_r->point())
                                                        * R_crcl
                                                        * obs_l->point();
                    double theta_lr = theta_lr_vector.squaredNorm();
                    // calculate matrix B [rbase_view_id]
                    // 对应公(18) 也能看出来变量里把transpose省略掉了，B对应right view的全局t
                    Eigen::Matrix3d Coefficient_B =
                            xi_cross * R_cicl * obs_l->point() * a_lr_t *
                            frame_rot.at(rbase_view_id).transpose();
                    // calculate matrix C [i_view_id]
                    Eigen::Matrix3d Coefficient_C = theta_lr * cross_product_matrix(obs_i->point()) *
                                                    frame_rot.at(i_view_id).transpose();
                    // calculate matrix D [lbase_view_id]
                    Eigen::Matrix3d Coefficient_D = -(Coefficient_B + Coefficient_C);
                    // calculate temp matrix L for a single 3D matrix
                    tmp_LiGT_vec.setZero();
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(rbase_view_id) * 3) += Coefficient_B;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(i_view_id) * 3) += Coefficient_C;
                    tmp_LiGT_vec.block<3, 3>(0, time_frameid2_int_frameid.at(lbase_view_id) * 3) += Coefficient_D;
                    // calculate LtL submodule
                    Eigen::MatrixXd LTL_l_row = Coefficient_D.transpose() * tmp_LiGT_vec * all_weights[track_id];
                    Eigen::MatrixXd LTL_r_row = Coefficient_B.transpose() * tmp_LiGT_vec * all_weights[track_id];
                    Eigen::MatrixXd LTL_i_row = Coefficient_C.transpose() * tmp_LiGT_vec * all_weights[track_id];
                    // assignment for LtL (except for the reference view id)
                    // #pragma omp critical
                    {
                        if (time_frameid2_int_frameid.at(lbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(lbase_view_id) * 3 - 3) += LTL_l_row.rightCols(
                                    LTL_l_row.cols() - 3);
                        if (time_frameid2_int_frameid.at(rbase_view_id) > 0)
                            LTL.middleRows<3>(
                                    time_frameid2_int_frameid.at(rbase_view_id) * 3 - 3) += LTL_r_row.rightCols(
                                    LTL_r_row.cols() - 3);
                        if (time_frameid2_int_frameid.at(i_view_id) > 0)
                            LTL.middleRows<3>(time_frameid2_int_frameid.at(i_view_id) * 3 - 3) += LTL_i_row.rightCols(
                                    LTL_i_row.cols() - 3);
                    }
                }
            }
            ++track_id;
        }
    }
    std::cout << "build_LTL ids: " << track_id << std::endl;
}

void GVINS::select_base_views(const Eigen::aligned_map<double, Eigen::Matrix3d> &frame_rot, const Eigen::aligned_map<double, FeaturePerFrame> &track, 
                            const std::map<int, double> &int_frameid2_time_frameid, const std::map<double, int> &time_frameid2_int_frameid, 
                            double &lbase_view_id, double &rbase_view_id) {
    double best_criterion_value = -1.;
    std::vector<int> track_id;

    for (const auto &frame: track) {
        int id = time_frameid2_int_frameid.at(frame.first);
        track_id.push_back(id);
    }

    size_t track_size = track_id.size(); //num_pts_

    // [Step.2 in Pose-only Algorithm]: select the left/right-base views
    for (size_t i = 0; i < track_size - 1; ++i) {
        for (size_t j = i + 1; j < track_size; ++j) {

            const double &i_view_id = int_frameid2_time_frameid.at(track_id[i]);
            const double &j_view_id = int_frameid2_time_frameid.at(track_id[j]);

            const Eigen::Vector3d &i_coord = track.at(i_view_id).point();
            const Eigen::Vector3d &j_coord = track.at(j_view_id).point();

            // R_i is world to camera i
            const Eigen::Matrix3d &R_i = frame_rot.at(i_view_id);
            const Eigen::Matrix3d &R_j = frame_rot.at(j_view_id);
            // camera i to camera j
            // Rcjw *  Rwci
            const Eigen::Matrix3d R_ij = R_j.transpose() * R_i;
            const Eigen::Vector3d theta_ij = j_coord.cross(R_ij * i_coord);

            double criterion_value = theta_ij.norm();

            if (criterion_value > best_criterion_value) {

                best_criterion_value = criterion_value;

                if (i_view_id < j_view_id) {
                    lbase_view_id = i_view_id;
                    rbase_view_id = j_view_id;
                } else {
                    lbase_view_id = j_view_id;
                    rbase_view_id = i_view_id;
                }
            }
        }
    }
}

void GVINS::select_base_views(const Eigen::aligned_map<double, Eigen::Matrix3d> &frame_rot, const Eigen::aligned_unordered_map<double, std::weak_ptr<Feature>> &track, 
                            const std::map<int, double> &int_frameid2_time_frameid, const std::map<double, int> &time_frameid2_int_frameid, 
                            double &lbase_view_id, double &rbase_view_id) {
    double best_criterion_value = -1.;
    std::vector<int> track_id;

    for (const auto &frame: track) {
        int id = time_frameid2_int_frameid.at(frame.first);
        track_id.push_back(id);
    }

    size_t track_size = track_id.size(); //num_pts_

    // [Step.2 in Pose-only Algorithm]: select the left/right-base views
    for (size_t i = 0; i < track_size - 1; ++i) {
        for (size_t j = i + 1; j < track_size; ++j) {

            const double &i_view_id = int_frameid2_time_frameid.at(track_id[i]);
            const double &j_view_id = int_frameid2_time_frameid.at(track_id[j]);

            auto feature1 = track.at(i_view_id).lock();
            auto feature2 = track.at(j_view_id).lock();

            const Eigen::Vector3d &i_coord = feature1->point();
            const Eigen::Vector3d &j_coord = feature2->point();

            // R_i is world to camera i
            const Eigen::Matrix3d &R_i = frame_rot.at(i_view_id);
            const Eigen::Matrix3d &R_j = frame_rot.at(j_view_id);
            // camera i to camera j
            // Rcjw *  Rwci
            const Eigen::Matrix3d R_ij = R_j.transpose() * R_i;
            const Eigen::Vector3d theta_ij = j_coord.cross(R_ij * i_coord);

            double criterion_value = theta_ij.norm();

            if (criterion_value > best_criterion_value) {

                best_criterion_value = criterion_value;

                if (i_view_id < j_view_id) {
                    lbase_view_id = i_view_id;
                    rbase_view_id = j_view_id;
                } else {
                    lbase_view_id = j_view_id;
                    rbase_view_id = i_view_id;
                }
            }
        }
    }
}

bool GVINS::solve_LTL(const Eigen::MatrixXd &LTL, Eigen::VectorXd &evectors) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(LTL);
    if (es.info() != Eigen::Success) {
        std::cout << "SelfAdjointEigenSolver failed in solve_LTL" << std::endl;
        return false;
    }

    // ============ 新增：稳定性检查 ============
    Eigen::VectorXd eigenvalues = es.eigenvalues(); // 按 Eigen 文档是升序（最小到最大）
    const double epsilon = 1e-8; // 非零判定阈值（可按需要放宽到 1e-8 等）
    int n = static_cast<int>(eigenvalues.size());

    // TODO 1: 找到第一个“非零”特征值的索引（第一个 > epsilon）
    int first_nonzero_idx = -1;
    {
        for (int i = 0; i < n; ++i) {
            if (eigenvalues(i) > epsilon) {
                first_nonzero_idx = i;
                break;
            }
        }

        if (first_nonzero_idx == -1) {
            LOGW << "All eigenvalues are ~zero (or <= epsilon). LTL is singular.";
            return false;
        }

        // TODO 2: 检查秩（至少需要多少个非零特征值？）
        int rank = n - first_nonzero_idx;

        // 允许的零特征数量（典型情况下可留出少量自由度/gauge freedom）
        const int allowed_nulls = 3; // 可调：如果你认为可能只有 1 或 2 个自由度，改成相应值
        int min_required_rank = std::max(1, n - allowed_nulls);

        if (rank < min_required_rank) {
            LOGW << "LTL rank deficient: rank = " << rank
                << ", required >= " << min_required_rank;
            return false;
        }

        // TODO 3: 计算条件数
        double lambda_min = eigenvalues(first_nonzero_idx);
        double lambda_max = eigenvalues(n - 1);

        if (lambda_min <= 0.0) {
            LOGW << "Non-positive minimal eigenvalue: " << lambda_min;
            return false;
        }

        // 我们用 LTL 的条件数 lambda_max / lambda_min（也可以使用 sqrt 版，根据你需要）
        double condition_number = std::sqrt(lambda_max / lambda_min);

        LOGW << "LTL condition number (lambda_max / lambda_min) = " << condition_number
            << ", lambda_min = " << lambda_min << ", lambda_max = " << lambda_max;

        // TODO 4: 调用稳定性检查
        if (!checkHessianStability(condition_number)) {
            LOGW << "Hessian stability check failed.";
            return false;
        }
    }

    int dim = LTL.rows();        // = 3*num_view - 3
    int total = dim + 3;         // = 3*num_view

    evectors.resize(total);
    evectors.setZero();

    Eigen::VectorXd x = es.eigenvectors().col(first_nonzero_idx);//0

    evectors.tail(dim) = x;

    return true;
}

// 重载版本：直接接受条件数的稳定性检查实现示例
bool GVINS::checkHessianStability(double condition_number) {
    // 1. 条件数是否过大
    if (condition_number > 1e5) {
        LOGW << "Condition number too large: " << condition_number;
        last_condition_number_ = 0.0;
        stable_check_count_ = 0;
        return false;
    }

    // 2. 相对变化检查
    if (last_condition_number_ > 0.0) {
        double relative_change = std::abs(condition_number - last_condition_number_) / last_condition_number_;
        LOGW << "Condition number relative change: " << (relative_change * 100.0) << "%";

        if (relative_change < 0.5) {
            stable_check_count_++;
        } else {
            stable_check_count_ = 0;
        }
    } else {
        // 第一次测得，初始化计数
        stable_check_count_ = 0;
    }

    last_condition_number_ = condition_number;

    bool is_stable = (stable_check_count_ >= 2);
    if (is_stable) {
        LOGW << "Hessian stable after " << stable_check_count_ << " checks";
        // 你可以在这里重置计数或保留上一次值以便后续检测
        last_condition_number_ = 0.0;
        stable_check_count_ = 0;
    }

    return is_stable;
}

void GVINS::identify_sign(const Eigen::MatrixXd &A_lr, Eigen::VectorXd &evectors) {
    const Eigen::VectorXd judgeValue = A_lr * evectors;
    const int positive_count = (judgeValue.array() > 0.0).cast<int>().sum();
    const int negative_count = judgeValue.rows() - positive_count;
    if (positive_count < negative_count) {
        evectors = -evectors;
    }
    // std::cout << "positive_count: " << positive_count << "  negative_count: " << negative_count << std::endl;
}

bool GVINS::linearAlignment(const std::map<int, double> &int_frameid2_time_frameid, const std::map<double, int> &time_frameid2_int_frameid, 
                            std::vector<Eigen::Vector3d> &velocity, std::vector<Eigen::Vector3d> &position, std::vector<Eigen::Matrix3d> &rotation) {
    int all_frame_count = int_frameid2_time_frameid.size();
    int n_state = all_frame_count * 3 + 3 + 1;
    auto G = integration_config_.gravity;
    Vector3d g;
    Eigen::Vector3d pose_b_c_t = pose_b_c_.t;
    Eigen::Matrix3d Rs0 = Eigen::Quaterniond(statedatalist_.at(0).pose[6], statedatalist_.at(0).pose[3],
                                             statedatalist_.at(0).pose[4], statedatalist_.at(0).pose[5]).toRotationMatrix();
    
    Eigen::IOFormat LogFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", "");

    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();
    double Q = 0.;

    for (size_t i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
        size_t j = i + 1;
        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        std::shared_ptr<PreintegrationBase> imu1 = preintegrationlist_.at(i);

        double dt = imu1->deltaTime();

        tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 6) = rotation[i].transpose() * (position[j] - position[i]) / 100.0;

        tmp_A.block<3, 3>(0, 7) = rotation[i].transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity() * G.norm();

        tmp_b.block<3, 1>(0, 0) = imu1->delta_p() + rotation[i].transpose() * rotation[j] * pose_b_c_t - pose_b_c_t;

        tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = rotation[i].transpose() * rotation[j];

        tmp_A.block<3, 3>(3, 7) = rotation[i].transpose() * dt * Eigen::Matrix3d::Identity() * G.norm();

        tmp_b.block<3, 1>(3, 0) = imu1->delta_v();

        Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();

        cov_inv.setIdentity();

        Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();

        Q += tmp_b.transpose() * cov_inv * tmp_b;
    }

    Eigen::VectorXd x;
    double s;
    Eigen::MatrixXd M_k2TM_k2 = A.bottomRightCorner<3, 3>();

    double mean_value = (M_k2TM_k2(0, 0) + M_k2TM_k2(1, 1) + M_k2TM_k2(2, 2)) / 3.0;

    double scale = 1 / mean_value;
    A = A * scale;
    b = b * scale;
    Q = Q * scale;

    if (!gravityRefine(A, -2. * b, Q, 1, x)) return false;

    g = x.tail(3) * G.norm();
    s = x(n_state - 4) / 100.0;
    x(n_state - 4) = s;

    LOGW << "scale: " << scale << ", s: " << s << ", g: " << g.format(LogFmt);

    for (int i = int_frameid2_time_frameid.size() - 1; i >= 0; i--) {
        position[i] = s * position[i] - rotation[i] * pose_b_c_t;
        // position[i] = s * position[i] - rotation[i] * tic[0] - (s * position[0] - rotation[0] * tic[0]);
        velocity[i] = rotation[i] * x.segment<3>(i * 3);
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (size_t i = 0; i < timelist_.size(); i++)
    {
        int id = time_frameid2_int_frameid.at(timelist_.at(i));
        position[id] = rot_diff * position[id];
        rotation[id] = rot_diff * rotation[id];
        velocity[id] = rot_diff * velocity[id];
    }
    LOGW << "g0     " << g.format(LogFmt);
    LOGW << "my R0  " << Utility::R2ypr(Rs0).format(LogFmt);

    // Eigen::Vector3d biasg = Eigen::Map<const Eigen::Vector3d>(statedatalist_[0].mix + 3);
    // Eigen::Vector3d biasw(0, 0, 0);
    std::deque<std::pair<double, Pose>> Poselist_;
    {    
        for (size_t i = 0; i < int_frameid2_time_frameid.size() - 1; i++)
        {
            memcpy(statedatalist_.at(i).pose, position[i].data(), sizeof(double) * 3);
            memcpy(statedatalist_.at(i).pose + 3, Eigen::Quaterniond(rotation[i]).coeffs().data(), sizeof(double) * 4);
            memcpy(statedatalist_.at(i).mix, velocity[i].data(), sizeof(double) * 3);
            auto state = Preintegration::stateFromData(statedatalist_.at(i), preintegration_options_);
            Pose pose = MISC::stateToCameraPose(state, pose_b_c_);
            Poselist_.emplace_back(state.time, pose);
            preintegrationlist_.at(i)->reintegration(state);
        }
        memcpy(statedatalist_.at(int_frameid2_time_frameid.size() - 1).pose, position[int_frameid2_time_frameid.size() - 1].data(), sizeof(double) * 3);
        memcpy(statedatalist_.at(int_frameid2_time_frameid.size() - 1).pose + 3, Eigen::Quaterniond(rotation[int_frameid2_time_frameid.size() - 1]).coeffs().data(), sizeof(double) * 4);
        memcpy(statedatalist_.at(int_frameid2_time_frameid.size() - 1).mix, velocity[int_frameid2_time_frameid.size() - 1].data(), sizeof(double) * 3);
        auto state = Preintegration::stateFromData(statedatalist_.at(int_frameid2_time_frameid.size() - 1), preintegration_options_);
        Pose pose = MISC::stateToCameraPose(state, pose_b_c_);
        Poselist_.emplace_back(state.time, pose);
    }

    map_->retriangulate(Poselist_);
    // update statedatalist_
    // {    
    //     for (size_t i = 0; i < int_frameid2_time_frameid.size() - 1; i++)
    //     {
    //         double time0 = int_frameid2_time_frameid.at(i);
    //         auto frameptr = map_->getframebytime(time0);
    //         Pose pose0 = frameptr->pose();
    //         Pose pose_state;
    //         pose_state.R = pose0.R * pose_b_c_.R.transpose();
    //         pose_state.t = pose0.t - pose_state.R * pose_b_c_.t;
    //         memcpy(statedatalist_.at(i).pose, pose_state.t.data(), sizeof(double) * 3);
    //         memcpy(statedatalist_.at(i).pose + 3, Eigen::Quaterniond(pose_state.R).coeffs().data(), sizeof(double) * 4);
    //         auto state = Preintegration::stateFromData(statedatalist_.at(i), preintegration_options_);
    //         preintegrationlist_.at(i)->reintegration(state);
    //     }
    //     double time0 = int_frameid2_time_frameid.at(int_frameid2_time_frameid.size() - 1);
    //     auto frameptr = map_->getframebytime(time0);
    //     Pose pose0 = frameptr->pose();
    //     Pose pose_state;
    //     pose_state.R = pose0.R * pose_b_c_.R.transpose();
    //     pose_state.t = pose0.t - pose_state.R * pose_b_c_.t;
    //     memcpy(statedatalist_.at(int_frameid2_time_frameid.size() - 1).pose, pose_state.t.data(), sizeof(double) * 3);
    //     memcpy(statedatalist_.at(int_frameid2_time_frameid.size() - 1).pose + 3, Eigen::Quaterniond(pose_state.R).coeffs().data(), sizeof(double) * 4);
    //     auto state = Preintegration::stateFromData(statedatalist_.at(int_frameid2_time_frameid.size() - 1), preintegration_options_);
    // }

    return true;
}

bool GVINS::gravityRefine(const Eigen::MatrixXd &M,
                                const Eigen::VectorXd &m,
                                double Q,
                                double gravity_mag,
                                Eigen::VectorXd &rhs) {

    // // Solve
    // int q = M.rows() - 3;

    // Eigen::MatrixXd A = 2. * M.block(0, 0, q, q);
    // //LOG(INFO) << StringPrintf("A: %.16f", A);

    // // TODO Check if A is invertible!!
    // //Eigen::Matrix3d A_ = A.block<3, 3>(1, 1);
    // //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> svdA_(A_, Eigen::EigenvaluesOnly);
    // //result.svA_ = svdA_.eigenvalues();
    // //result.detA_ = A_.determinant();

    // Eigen::MatrixXd Bt = 2. * M.block(q, 0, 3, q);
    // Eigen::MatrixXd BtAi = Bt * A.inverse();

    // Eigen::Matrix3d D = 2. * M.block(q, q, 3, 3);
    // Eigen::Matrix3d S = D - BtAi * Bt.transpose();

    // // TODO Check if S is invertible!
    // //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> svdS(S, Eigen::EigenvaluesOnly);
    // //result.svS = svdS.eigenvalues();
    // //result.detS = S.determinant();
    // //LOG(INFO) << StringPrintf("det(S): %.16f", S.determinant());
    // //LOG(INFO) << StringPrintf("eigenvalues(S): %.16f %.16f %.16f",
    // //                          c[0], svd.eigenvalues()[1], svd.eigenvalues()[2]);

    // Eigen::Matrix3d Sa = S.determinant() * S.inverse();
    // Eigen::Matrix3d U = S.trace() * Eigen::Matrix3d::Identity() - S;

    // Eigen::Vector3d v1 = BtAi * m.head(q);
    // Eigen::Vector3d m2 = m.tail<3>();

    // Eigen::Matrix3d X;
    // Eigen::Vector3d Xm2;

    // // X = I
    // const double c4 = 16. * (v1.dot(v1) - 2. * v1.dot(m2) + m2.dot(m2));

    // X = U;
    // Xm2 = X * m2;
    // const double c3 = 16. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

    // X = 2. * Sa + U * U;
    // Xm2 = X * m2;
    // const double c2 = 4. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

    // X = Sa * U + U * Sa;
    // Xm2 = X * m2;
    // const double c1 = 2. * (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

    // X = Sa * Sa;
    // Xm2 = X * m2;
    // const double c0 = (v1.dot(X * v1) - 2. * v1.dot(Xm2) + m2.dot(Xm2));

    // const double s00 = S(0, 0), s01 = S(0, 1), s02 = S(0, 2);
    // const double s11 = S(1, 1), s12 = S(1, 2), s22 = S(2, 2);

    // const double t1 = s00 + s11 + s22;
    // const double t2 = s00 * s11 + s00 * s22 + s11 * s22
    //                     - std::pow(s01, 2) - std::pow(s02, 2) - std::pow(s12, 2);
    // const double t3 = s00 * s11 * s22 + 2. * s01 * s02 * s12
    //                     - s00 * std::pow(s12, 2) - s11 * std::pow(s02, 2) - s22 * std::pow(s01, 2);

    // Eigen::VectorXd coeffs(7);
    // coeffs << 64.,
    //         64. * t1,
    //         16. * (std::pow(t1, 2) + 2. * t2),
    //         16. * (t1 * t2 + t3),
    //         4. * (std::pow(t2, 2) + 2. * t1 * t3),
    //         4. * t3 * t2,
    //         std::pow(t3, 2);

    // const double G2i = 1. / std::pow(gravity_mag, 2);

    // coeffs(2) -= c4 * G2i;
    // coeffs(3) -= c3 * G2i;
    // coeffs(4) -= c2 * G2i;
    // coeffs(5) -= c1 * G2i;
    // coeffs(6) -= c0 * G2i;

    // Eigen::VectorXd real, imag;
    // if (!FindPolynomialRootsCompanionMatrix(coeffs, &real, &imag)) {
    //     LOG(ERROR) << "Failed to find roots\n";
    //     printf("%.16f %.16f %.16f %.16f %.16f %.16f %.16f",
    //             coeffs[0], coeffs[1], coeffs[2], coeffs[3],
    //             coeffs[4], coeffs[5], coeffs[6]);

    //     return false;
    // }

    // Eigen::VectorXd lambdas = real_roots(real, imag);
    // if (lambdas.size() == 0) {
    //     LOG(ERROR) << "No real roots found\n";
    //     printf("%.16f %.16f %.16f %.16f %.16f %.16f %.16f",
    //             coeffs[0], coeffs[1], coeffs[2], coeffs[3],
    //             coeffs[4], coeffs[5], coeffs[6]);

    //     return false;
    // }

    // Eigen::MatrixXd W(M.rows(), M.rows());
    // W.setZero();
    // W.block<3, 3>(q, q) = Eigen::Matrix3d::Identity();

    // Eigen::VectorXd solution;
    // double min_cost = std::numeric_limits<double>::max();
    // for (Eigen::VectorXd::Index i = 0; i < lambdas.size(); ++i) {
    //     const double lambda = lambdas(i);

    //     Eigen::FullPivLU<Eigen::MatrixXd> lu(2. * M + 2. * lambda * W);
    //     Eigen::VectorXd x_ = -lu.inverse() * m;

    //     double cost = x_.transpose() * M * x_;
    //     cost += m.transpose() * x_;
    //     cost += Q;

    //     if (cost < min_cost) {
    //         solution = x_;
    //         min_cost = cost;
    //     }
    // }


    // const double constraint = solution.transpose() * W * solution;

    //    if (solution[0] < 1e-3 || constraint < 0.
    //        || std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag > 1e-3) { // TODO
    //        LOG(WARNING) << "Discarding bad solution...\n";
    //        printf("constraint: %.16f\n", constraint);
    //        printf("constraint error: %.2f\n",
    //               100. * std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag);
    //        return false;
    //    }

    // // if (constraint < 0.
    // //     || std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag > 1e-3) { // TODO
    // //     LOG(WARNING) << "Discarding bad solution...\n";
    // //     printf("constraint: %.16f\n", constraint);
    // //     printf("constraint error: %.2f\n",
    // //             100. * std::abs(std::sqrt(constraint) - gravity_mag) / gravity_mag);
    // //     return false;
    // // }

    // rhs = solution;

    // return true;

        // M: (q+3) x (q+3), 结构为 [A_bt; Bt^T D] 其中 q = M.rows() - 3
    // m: vector length q+3
    // 目标：在对速度等变量消元后，解一个带重力模长约束的二次型最小化问题
    // 返回 rhs 为完整的解向量 x

    // q 为非重力部分的维度
    const int q = M.rows() - 3;
    if (q <= 0) return false;

    // A = 2 * M(0:q,0:q)
    Eigen::MatrixXd A = 2.0 * M.block(0, 0, q, q);

    // Bt = 2 * M(q:q+3, 0:q)  (3 x q)
    Eigen::MatrixXd Bt = 2.0 * M.block(q, 0, 3, q);

    // 为避免显式求逆，使用分解器求 BtAi = Bt * A^{-1}
    // 我们先尝试 A 的 LDLT（针对对称正定或半正定），失败则退回 FullPivLU
    Eigen::MatrixXd BtAi; // 3 x q
    {
        Eigen::LDLT<Eigen::MatrixXd> ldA(A);
        if (ldA.info() == Eigen::Success) {
            // 计算 temp = A^{-1} * Bt^T  (q x 3)，然后转置得到 BtAi (3 x q)
            Eigen::MatrixXd temp = ldA.solve(Bt.transpose()); // q x 3
            BtAi = temp.transpose(); // 3 x q
        } else {
            // LDLT 失败，尝试更稳健的 LU 分解
            Eigen::FullPivLU<Eigen::MatrixXd> luA(A);
            if (!luA.isInvertible()) {
                // A 无法逆（信息不足），退化处理：返回失败
                return false;
            }
            Eigen::MatrixXd temp = luA.solve(Bt.transpose()); // q x 3
            BtAi = temp.transpose(); // 3 x q
        }
    }

    // D = 2 * M(q:q+3, q:q+3) (3x3)
    Eigen::Matrix3d D = 2.0 * M.block(q, q, 3, 3);

    // S = D - BtAi * Bt^T  （3x3 Schur 补）
    Eigen::Matrix3d S = D - BtAi * Bt.transpose();

    // 对 S 做稳健处理：若接近奇异，进行小幅正则化
    double detS = S.determinant();
    Eigen::Matrix3d Sa; // = det(S) * S^{-1}，但对奇异情形要保护
    const double EPS_S_DET = 1e-12;
    if (std::abs(detS) < EPS_S_DET) {
        // 正则化 S 并计算 Sa
        const double reg = 1e-8; // 可根据需要微调
        Eigen::Matrix3d Sreg = S + reg * Eigen::Matrix3d::Identity();
        double detSreg = Sreg.determinant();
        if (std::abs(detSreg) < 1e-20) {
            // 即使正则化后仍然异常，放弃求解
            return false;
        }
        Sa = detSreg * Sreg.inverse();
    } else {
        Sa = detS * S.inverse();
    }

    // U = trace(S) * I - S
    Eigen::Matrix3d U = S.trace() * Eigen::Matrix3d::Identity() - S;

    // 计算 v1 = BtAi * m_head_q，m2 = m_tail_3
    Eigen::Vector3d v1 = BtAi * m.head(q);
    Eigen::Vector3d m2 = m.tail<3>();

    // 为构造多项式系数，逐项计算 X 与对应的二次型项
    // 避免重复内存分配，复用 X, Xm2
    Eigen::Matrix3d X;
    Eigen::Vector3d Xm2;

    // 常数项相关的 c4..c0（与原实现保持一致的表达）
    // c4
    const double c4 = 16.0 * (v1.dot(v1) - 2.0 * v1.dot(m2) + m2.dot(m2));

    // X = U
    X = U;
    Xm2 = X * m2;
    const double c3 = 16.0 * (v1.dot(X * v1) - 2.0 * v1.dot(Xm2) + m2.dot(Xm2));

    // X = 2 * Sa + U * U
    X = 2.0 * Sa + U * U;
    Xm2 = X * m2;
    const double c2 = 4.0 * (v1.dot(X * v1) - 2.0 * v1.dot(Xm2) + m2.dot(Xm2));

    // X = Sa * U + U * Sa
    X = Sa * U + U * Sa;
    Xm2 = X * m2;
    const double c1 = 2.0 * (v1.dot(X * v1) - 2.0 * v1.dot(Xm2) + m2.dot(Xm2));

    // X = Sa * Sa
    X = Sa * Sa;
    Xm2 = X * m2;
    const double c0 = (v1.dot(X * v1) - 2.0 * v1.dot(Xm2) + m2.dot(Xm2));

    // S 分量简要取值（用于构造多项式系数）
    const double s00 = S(0, 0), s01 = S(0, 1), s02 = S(0, 2);
    const double s11 = S(1, 1), s12 = S(1, 2), s22 = S(2, 2);

    // 计算 t1, t2, t3（3x3 矩阵 S 的特征多项式系数）
    const double t1 = s00 + s11 + s22;
    const double t2 = s00 * s11 + s00 * s22 + s11 * s22
                      - std::pow(s01, 2) - std::pow(s02, 2) - std::pow(s12, 2);
    const double t3 = s00 * s11 * s22 + 2.0 * s01 * s02 * s12
                      - s00 * std::pow(s12, 2) - s11 * std::pow(s02, 2) - s22 * std::pow(s01, 2);

    // 构造 7 阶多项式系数（从高阶到常数项）
    Eigen::VectorXd coeffs(7);
    coeffs << 64.0,
              64.0 * t1,
              16.0 * (std::pow(t1, 2) + 2.0 * t2),
              16.0 * (t1 * t2 + t3),
              4.0 * (std::pow(t2, 2) + 2.0 * t1 * t3),
              4.0 * t3 * t2,
              std::pow(t3, 2);

    // 因为多项式右侧减去了与重力模长有关的项，按原实现做修正
    const double G2i = 1.0 / (gravity_mag * gravity_mag);
    coeffs(2) -= c4 * G2i;
    coeffs(3) -= c3 * G2i;
    coeffs(4) -= c2 * G2i;
    coeffs(5) -= c1 * G2i;
    coeffs(6) -= c0 * G2i;

    // 求多项式根（复数形式），然后从中筛选实根
    Eigen::VectorXd real_parts, imag_parts;
    if (!FindPolynomialRootsCompanionMatrix(coeffs, &real_parts, &imag_parts)) {
        // 求根失败
        return false;
    }

    // 将实部/虚部分离并筛选可接受的实根（虚部足够小且为有限值）
    std::vector<double> lambdas;
    const double IMAG_TOL = 1e-8;
    for (int idx = 0; idx < real_parts.size(); ++idx) {
        double r = real_parts(idx), im = imag_parts(idx);
        if (!std::isfinite(r) || !std::isfinite(im)) continue;
        if (std::abs(im) <= IMAG_TOL) {
            lambdas.push_back(r);
        }
    }
    if (lambdas.empty()) {
        // 没有合格的实根
        return false;
    }

    // 构造 W：在 g 部分放 3x3 单位，其它位置为 0，W 大小为 M.rows()
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(M.rows(), M.rows());
    W.block<3, 3>(q, q) = Eigen::Matrix3d::Identity();

    // 在 lambdas 集合中逐个尝试，解 (2M + 2*lambda*W) * x = -m，选择使代价最小的解
    Eigen::VectorXd best_solution;
    double min_cost = std::numeric_limits<double>::infinity();
    for (double lambda : lambdas) {
        // K = 2*M + 2*lambda*W （对称矩阵）
        Eigen::MatrixXd K = 2.0 * M + 2.0 * lambda * W;

        // 尝试使用 LDLT（对称情况下通常更稳健），若失败则退到 FullPivLU
        Eigen::LDLT<Eigen::MatrixXd> ldK(K);
        Eigen::VectorXd x_candidate;
        bool solved = false;

        if (ldK.info() == Eigen::Success) {
            x_candidate = -ldK.solve(m);
            solved = true;
        } else {
            Eigen::FullPivLU<Eigen::MatrixXd> luK(K);
            if (luK.isInvertible()) {
                x_candidate = -luK.solve(m);
                solved = true;
            } else {
                // 无法解该 lambda，跳过
                solved = false;
            }
        }

        if (!solved) continue;

        // 计算代价 cost = x^T M x + m^T x + Q（这是原来代码使用的能量/代价函数）
        double cost = x_candidate.transpose() * M * x_candidate;
        cost += m.transpose() * x_candidate;
        cost += Q;

        if (!std::isfinite(cost)) continue;

        if (cost < min_cost) {
            min_cost = cost;
            best_solution = x_candidate;
        }
    }

    // 若没有找到任何可行解，则失败
    if (min_cost == std::numeric_limits<double>::infinity()) {
        return false;
    }

    // 检查并修正数值误差导致的微小负约束值
    double constraint = best_solution.transpose() * W * best_solution;
    if (constraint < -1e-9) {
        // 约束明显为负（数值或稳定性问题），视为失败
        return false;
    }
    // 取非负值用于模长比较
    double gnorm = std::sqrt(std::max(0.0, constraint));

    // 检验重力模长是否与期望一致（相对误差阈值可调整）
    const double REL_TOL = 1e-2; // 允许 1% 的相对误差（比原实现稍宽松，可按需调整）
    if (std::abs(gnorm - gravity_mag) / gravity_mag > REL_TOL) {
        return false;
    }

    // 最终结果写回 rhs
    rhs = best_solution;
    return true;

}

void GVINS::addNewKeyFrameTimeNode() {

    Lock lock(keyframes_mutex_);
    //tmp change
    Frame::Ptr frame0;
    while (!keyframes_.empty()) {
        // 取出一个关键帧
        // Obtain a new valid keyframe
        frame0       = keyframes_.front();
        double frametime = frame0->stamp();
        if (frametime > ins_window_.back().first.time) {
            break;
        }

        keyframes_.pop_front();

        lock.unlock();

        // 添加关键帧
        // Add new keyframe time node
        LOGI << "Insert keyframe " << frame0->keyFrameId() << " at " << Logging::doubleData(frame0->stamp()) << " with "
             << frame0->unupdatedMappoints().size() << " new mappoints";
        map_->insertKeyFrame(frame0);

        addNewTimeNode(frametime);
        // LOGI << "Add new keyframe time node at " << Logging::doubleData(frametime);

        lock.lock();
    }

    // 移除多余的预积分节点
    // Remove unused time node
    removeUnusedTimeNode();

    // if (((frame0->keyFrameState() == KEYFRAME_REMOVE_OLDEST || frame0->keyFrameState() == KEYFRAME_NORMAL) && !sub_statedatalist_.empty()) || 
    //     sub_statedatalist_.size() > 7) {
    //     map_->moveNewestKeyFrameToNonKeyFrame(frame0);
    //     sub_statedatalist_.push_back(statedatalist_.back());
    //     sub_timelist_.push_back(timelist_.back());
    //     gvinsSubOptimization();
    //     gvinsIncrementalMarginalization();
    //     sub_preintegrationlist_.clear();
    //     sub_statedatalist_.clear();
    //     sub_timelist_.clear();
    //     map_->clearSubMap();
    // }
}

bool GVINS::removeUnusedTimeNode() {
    if (unused_time_nodes_.empty()) {
        return false;
    }

    LOGI << "Remove " << unused_time_nodes_.size() << " unused time node "
         << Logging::doubleData(unused_time_nodes_[0]);

    for (double node : unused_time_nodes_) {
        int index = getStateDataIndex(node);

        // Exception
        if (index < 0) {
            continue;
        }

        auto first_preintegration  = preintegrationlist_[index - 1];
        auto second_preintegration = preintegrationlist_[index];
        auto imu_buffer            = second_preintegration->imuBuffer();

        // 将后一个预积分的IMU数据合并到前一个, 不包括第一个IMU数据
        // Merge the IMU preintegration
        for (size_t k = 1; k < imu_buffer.size(); k++) {
            first_preintegration->addNewImu(imu_buffer[k]);
        }

        // 移除时间节点, 以及后一个预积分
        // Remove the second time node
        preintegrationlist_.erase(preintegrationlist_.begin() + index);
        timelist_.erase(timelist_.begin() + index);
        statedatalist_.erase(statedatalist_.begin() + index);
    }
    unused_time_nodes_.clear();

    return true;
}

bool GVINS::removeUnusedTimeNode_old() {
    if (unused_time_nodes_.empty()) {
        return false;
    }

    LOGI << "Remove " << unused_time_nodes_.size() << " unused time node "
         << Logging::doubleData(unused_time_nodes_[0]);

    for (double node : unused_time_nodes_) {
        int index = getStateDataIndex(node);

        // Exception
        if (index < 0) {
            continue;
        }

        auto first_preintegration  = preintegrationlist_[index - 1];
        auto second_preintegration = preintegrationlist_[index];
        auto imu_buffer            = second_preintegration->imuBuffer();

        // 创建或者添加子图参数块
        if (sub_preintegrationlist_.empty()) {
            sub_preintegrationlist_.push_back(first_preintegration->clone());
            sub_preintegrationlist_.push_back(second_preintegration);
            sub_statedatalist_.push_back(statedatalist_[index - 1]);
            sub_statedatalist_.push_back(statedatalist_[index]);
            sub_timelist_.push_back(timelist_[index - 1]);
            sub_timelist_.push_back(timelist_[index]);
        } else {
            sub_preintegrationlist_.push_back(second_preintegration);
            sub_statedatalist_.push_back(statedatalist_[index]);
            sub_timelist_.push_back(timelist_[index]);
        }

        // 将后一个预积分的IMU数据合并到前一个, 不包括第一个IMU数据
        // Merge the IMU preintegration
        for (size_t k = 1; k < imu_buffer.size(); k++) {
            first_preintegration->addNewImu(imu_buffer[k]);
        }

        // 移除时间节点, 以及后一个预积分
        // Remove the second time node
        preintegrationlist_.erase(preintegrationlist_.begin() + index);
        timelist_.erase(timelist_.begin() + index);
        statedatalist_.erase(statedatalist_.begin() + index);
    }
    unused_time_nodes_.clear();

    return true;
}

void GVINS::addNewTimeNode(double time) {

    vector<IMU> series;
    IntegrationState state;

    // 获取时段内用于预积分的IMU数据
    // Obtain the IMU samples between the two time nodes
    double start = timelist_.back();
    double end   = time;
    MISC::getImuSeriesFromTo(ins_window_, start, end, series);

    state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);

    // 新建立新的预积分
    // Build a new IMU preintegration
    preintegrationlist_.emplace_back(
        Preintegration::createPreintegration(integration_parameters_, series[0], state, preintegration_options_));

    // 预积分, 从第二个历元开始
    // Add IMU sample
    for (size_t k = 1; k < series.size(); k++) {
        preintegrationlist_.back()->addNewImu(series[k]);
    }

    // 当前状态加入到滑窗中
    // Add current state and time node to the sliding window
    state      = preintegrationlist_.back()->currentState();
    state.time = time;

    statedatalist_.emplace_back(Preintegration::stateToData(state, preintegration_options_));
    timelist_.push_back(time);
}

void GVINS::parametersStatistic() {

    vector<double> parameters;

    // 所有关键帧
    // All keyframes
    vector<ulong> keyframeids = map_->orderedKeyFrames();
    size_t size               = keyframeids.size();
    if (size < 2) {
        return;
    }
    auto keyframes = map_->keyframes();

    // 最新的关键帧
    // The latest keyframe
    auto frame_cur = keyframes.at(keyframeids[size - 1]);
    auto frame_pre = keyframes.at(keyframeids[size - 2]);

    // 时间戳
    // Time stamp
    parameters.push_back(frame_cur->stamp());
    parameters.push_back(frame_cur->stamp() - frame_pre->stamp());

    // 当前关键帧与上一个关键帧的id差, 即最新关键帧的跟踪帧数
    // Interval
    auto frame_cnt = static_cast<double>(frame_cur->id() - frame_pre->id());
    parameters.push_back(frame_cnt);

    // 特征点数量
    // Feature points
    parameters.push_back(static_cast<double>(frame_cur->numFeatures()));

    // 路标点重投影误差统计
    // Reprojection errors
    vector<double> reprojection_errors;
    for (auto &landmark : map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        vector<double> errors;
        for (auto &observation : mappoint->observations()) {
            auto feat = observation.second.lock();
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            if (!frame || !frame->isKeyFrame() || !map_->isKeyFrameInMap(frame)) {
                continue;
            }

            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), feat->keyPoint()).norm();
            errors.push_back(error);
        }
        if (errors.empty()) {
            LOGE << "Mappoint " << mappoint->id() << " with zero observation";
            continue;
        }
        double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        reprojection_errors.emplace_back(avg_error);
    }

    if (reprojection_errors.empty()) {
        reprojection_errors.push_back(0);
    }

    double min_error = *std::min_element(reprojection_errors.begin(), reprojection_errors.end());
    parameters.push_back(min_error);
    double max_error = *std::max_element(reprojection_errors.begin(), reprojection_errors.end());
    parameters.push_back(max_error);
    double avg_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) /
                       static_cast<double>(reprojection_errors.size());
    parameters.push_back(avg_error);
    double sq_sum =
        std::inner_product(reprojection_errors.begin(), reprojection_errors.end(), reprojection_errors.begin(), 0.0);
    double rms_error = std::sqrt(sq_sum / static_cast<double>(reprojection_errors.size()));
    parameters.push_back(rms_error);

    // 迭代次数
    // Iterations
    parameters.push_back(iterations_[0]);
    parameters.push_back(iterations_[1]);

    // 计算耗时
    // Time cost
    parameters.push_back(timecosts_[0]);
    parameters.push_back(timecosts_[1]);
    parameters.push_back(timecosts_[2]);

    // 路标点粗差
    // Outliers
    parameters.push_back(outliers_[0]);
    parameters.push_back(outliers_[1]);

    // 保存数据
    // Dump current parameters
    statfilesaver_->dump(parameters);
    statfilesaver_->flush();
}

bool GVINS::gvinsOutlierCulling() {
    if (map_->keyframes().empty()) {
        return false;
    }

    // 移除非关键帧中的路标点, 不能在遍历中直接移除, 否则破坏了遍历
    // Find outliers first and remove later
    vector<MapPoint::Ptr> mappoints;
    int num_outliers_mappoint = 0;
    int num_outliers_feature  = 0;
    int num1 = 0, num2 = 0, num3 = 0;
    for (auto &landmark : map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        // 未参与优化的无效路标点
        // Only those in the sliding window
        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        // 路标点在滑动窗口内的所有观测
        // All the observations for mappoint
        vector<double> errors;
        for (auto &observation : mappoint->observations()) {
            auto feat = observation.second.lock();
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            if (!frame || !frame->isKeyFrame() || !map_->isKeyFrameInMap(frame)) {
                continue;
            }

            auto pp = feat->keyPoint();

            // 计算重投影误差
            // Calculate the reprojection error
            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), pp).norm();

            // 大于3倍阈值, 则禁用当前观测
            // Feature outlier
            if (!tracking_->isGoodToTrack(pp, frame->pose(), mappoint->pos(), 4.0)) {//4.0
                feat->setOutlier(true);
                mappoint->decreaseUsedTimes();

                // 如果当前观测帧是路标点的参考帧, 直接设置为outlier
                // Mappoint
                if (frame->id() == mappoint->referenceFrameId()) {
                    mappoint->setOutlier(true);
                    mappoints.push_back(mappoint);
                    num_outliers_mappoint++;
                    num1++;
                    break;
                }
                num_outliers_feature++;
            } else {
                errors.push_back(error);
            }
        }

        // 有效观测不足, 平均重投影误差较大, 则为粗差
        // Mappoint outlier
        if (errors.size() < 2) {
            mappoint->setOutlier(true);
            mappoints.push_back(mappoint);
            num_outliers_mappoint++;
            num2++;
        } else {
            double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
            if (avg_error > 2.5 * reprojection_error_std_) {
                mappoint->setOutlier(true);
                mappoints.push_back(mappoint);
                num_outliers_mappoint++;
                num3++;
            }
        }
    }

    // 移除outliers
    // Remove the mappoint outliers
    for (auto &mappoint : mappoints) {
        map_->removeMappoint(mappoint);
    }

    LOGI << "Culled " << num_outliers_mappoint << " mappoint with " << num_outliers_feature << " bad observed features "
         << num1 << ", " << num2 << ", " << num3;
    outliers_[0] = num_outliers_mappoint;
    outliers_[1] = num_outliers_feature;

    return true;
}

bool GVINS::gvinsOptimization() {
    static int first_num_iterations  = optimize_num_iterations_ / 4;
    static int second_num_iterations = optimize_num_iterations_ - first_num_iterations;

    TimeCost timecost;

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type         = ceres::DENSE_SCHUR;
    options.max_num_iterations         = first_num_iterations;
    options.num_threads                = 4;

    // 状态参数
    // State parameters
    addStateParameters(problem);

    // 重投影参数
    // Visual parameters
    addReprojectionParameters(problem);

    // 边缘化残差
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        auto factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(factor, nullptr, last_marginalization_parameter_blocks_);
    }

    // 预积分残差
    // The IMU preintegration factors
    // addImuFactors(problem);
    addImuFactors_old(problem);

    //tmp change

    // 视觉重投影残差
    // The visual reprojection factors
    auto residual_ids = addReprojectionFactors_old(problem, true);

    LOGI << "Add " << preintegrationlist_.size() << " preintegration, "
         << residual_ids.size() << " reprojection";

    // addVisualpriorFactor(problem);

    #pragma region 1
    {
        for (auto & state: statedatalist_) {
            std::cout << std::fixed << std::setprecision(9);
            for (int i = 0; i < 7; ++i) std::cout << state.pose[i] << " ";
            for (int i = 0; i < 9; ++i) std::cout << state.mix[i] << " ";
            std::cout << std::endl;
        }

        //imu
        double imu_total_squared_residual_sum = 0.0;
        std::vector<double> imu_individual_squared_sums;
        for (auto &id : imu_factor_blocks_) {
            // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
            double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
            problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
            // 保存每一维残差的平方
            std::vector<double> squared_components;
            // 计算该残差的平方和
            double squared_sum = 0.0;
            for (int i = 0; i < 15; ++i) {
                double sq = residuals[i] * residuals[i];
                squared_components.push_back(sq);
                squared_sum += sq;
            }
            imu_individual_squared_sums.push_back(squared_sum);
            imu_total_squared_residual_sum += squared_sum;

            LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
                                     << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
                                     << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
                                     << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
                                     << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
        }
        // 平均残差平方和
        double imu_average_squared_residual = 0.0;
        if (!imu_individual_squared_sums.empty()) {
            imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
        }
        // 打印结果
        LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
        LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
        // 可选：打印每一个残差平方和
        for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
            LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
        }

        // {
        //     double residuals[6] = {0};  // 可根据实际最大残差维度调整大小
        //     problem.EvaluateResidualBlock(imu_error_block_, false, nullptr, residuals, nullptr);
        //     // 计算该残差的平方和
        //     double squared_sum = 0.0;
        //     for (int i = 0; i < 6; ++i) {
        //         squared_sum += residuals[i] * residuals[i];
        //     }
        //     LOGI << "IMU error Total squared residual sum: " << squared_sum;
        // }

        //visual
        double total_squared_residual_sum = 0.0;
        size_t outlier_count = 0;
        std::vector<double> individual_squared_sums;
        for (auto &info : residual_ids) {
            auto &id = info.id;
            // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
            double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
            problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
            // 计算该残差的平方和
            double squared_sum = 0.0;
            for (int i = 0; i < 2; ++i) {
                squared_sum += residuals[i] * residuals[i];
            }
            individual_squared_sums.push_back(squared_sum);
            total_squared_residual_sum += squared_sum;
            if (squared_sum > 10.0) {
                outlier_count++;
            }
        }
        // 平均残差平方和
        double average_squared_residual = 0.0;
        if (!individual_squared_sums.empty()) {
            average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
        }
        // 打印结果
        LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
        LOGI << "Visual Average squared residual: " << average_squared_residual;
        LOGI << "Visual Number of residuals > 10: " << outlier_count;
        // // 可选：打印每一个残差平方和
        // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
        //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
        // }
    }
    #pragma endregion

    // 第一次优化
    // The first optimization
    {
        timecost.restart();

        solver.Solve(options, &problem, &summary);
        LOGI << summary.BriefReport();

        iterations_[0] = summary.num_successful_steps;
        timecosts_[0]  = timecost.costInMillisecond();

        // fixUnstableInverseDepths(problem);

        // LOGI << "iterations_: " << iterations_[0] << ", timecosts_: " << timecosts_[0];
    }

    // 粗差检测和剔除
    // Outlier detetion for GNSS and visual
    {
        // Remove factors in the final

        // Remove outlier reprojection factors
        // removeReprojectionFactorsByChi2_old(problem, residual_ids, 5.991);//5.991

    }

    #pragma region 2
    // {
    //     for (auto & state: statedatalist_) {
    //         std::cout << std::fixed << std::setprecision(9);
    //         for (int i = 0; i < 7; ++i) std::cout << state.pose[i] << " ";
    //         for (int i = 0; i < 9; ++i) std::cout << state.mix[i] << " ";
    //         std::cout << std::endl;
    //     }

    //     //imu
    //     double imu_total_squared_residual_sum = 0.0;
    //     std::vector<double> imu_individual_squared_sums;
    //     for (auto &id : imu_factor_blocks_) {
    //         // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 保存每一维残差的平方
    //         std::vector<double> squared_components;
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 15; ++i) {
    //             double sq = residuals[i] * residuals[i];
    //             squared_components.push_back(sq);
    //             squared_sum += sq;
    //         }
    //         imu_individual_squared_sums.push_back(squared_sum);
    //         imu_total_squared_residual_sum += squared_sum;

    //         LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
    //                                  << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
    //                                  << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
    //                                  << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
    //                                  << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
    //     }
    //     // 平均残差平方和
    //     double imu_average_squared_residual = 0.0;
    //     if (!imu_individual_squared_sums.empty()) {
    //         imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
    //     LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
    //     // 可选：打印每一个残差平方和
    //     for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
    //         LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
    //     }

    //     // {
    //     //     double residuals[6] = {0};  // 可根据实际最大残差维度调整大小
    //     //     problem.EvaluateResidualBlock(imu_error_block_, false, nullptr, residuals, nullptr);
    //     //     // 计算该残差的平方和
    //     //     double squared_sum = 0.0;
    //     //     for (int i = 0; i < 6; ++i) {
    //     //         squared_sum += residuals[i] * residuals[i];
    //     //     }
    //     //     LOGI << "IMU error Total squared residual sum: " << squared_sum;
    //     // }

    //     //visual
    //     double total_squared_residual_sum = 0.0;
    //     size_t outlier_count = 0;
    //     std::vector<double> individual_squared_sums;
    //     for (auto &info : residual_ids) {
    //         auto &id = info.id;
    //         // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 2; ++i) {
    //             squared_sum += residuals[i] * residuals[i];
    //         }
    //         individual_squared_sums.push_back(squared_sum);
    //         total_squared_residual_sum += squared_sum;
    //         if (squared_sum > 10.0) {
    //             outlier_count++;
    //         }
    //     }
    //     // 平均残差平方和
    //     double average_squared_residual = 0.0;
    //     if (!individual_squared_sums.empty()) {
    //         average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
    //     LOGI << "Visual Average squared residual: " << average_squared_residual;
    //     LOGI << "Visual Number of residuals > 10: " << outlier_count;
    //     // // 可选：打印每一个残差平方和
    //     // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
    //     //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
    //     // }
    // }
    #pragma endregion

    // 第二次优化
    // The second optimization
    {
        // options.max_num_iterations = second_num_iterations;

        // timecost.restart();

        // solver.Solve(options, &problem, &summary);
        // LOGI << summary.BriefReport();

        // iterations_[1] = summary.num_successful_steps;
        // timecosts_[1]  = timecost.costInMillisecond();

        // LOGI << "iterations_: " << iterations_[1] << ", timecosts_: " << timecosts_[1];

        if (!map_->isMaximumKeframes()) {
            // 进行必要的重积分
            // Reintegration during initialization
            doReintegration();
        }
    }

    // 更新参数, 必须的
    // Update the parameters from the optimizer
    updateParametersFromOptimizer();

    // 移除粗差路标点
    // Remove mappoint and feature outliers
    gvinsOutlierCulling();

    return true;
}

int GVINS::removeReprojectionFactorsByChi2(ceres::Problem &problem, vector<ceres::ResidualBlockId> &residual_ids,
                                           double chi2) {
    double cost;
    int outlier_features = 0;

    // 进行卡方检验, 判定粗差因子, 待全部判定完成再进行移除, 否则会导致错误
    // Judge first and remove later
    vector<ceres::ResidualBlockId> outlier_residual_ids;
    for (auto &id : residual_ids) {
        problem.EvaluateResidualBlock(id, false, &cost, nullptr, nullptr);

        // cost带有1/2系数
        // To chi2
        if (cost * 2.0 > chi2) {
            outlier_features++;
            outlier_residual_ids.push_back(id);
        }
    }

    // 从优化问题中移除所有粗差因子
    // Remove the outliers from the optimizer
    for (auto &id : outlier_residual_ids) {
        problem.RemoveResidualBlock(id);
    }

    LOGI << "Remove " << outlier_features << " reprojection factors";

    return outlier_features;
}

void GVINS::updateParametersFromOptimizer() {
    if (map_->keyframes().empty()) {
        return;
    }

    // 先更新外参, 更新位姿需要外参
    // Update the extrinsic first
    {
        if (optimize_estimate_td_) {
            td_b_c_ = extrinsic_[7];
            LOGI << "Update td to " << td_b_c_ ;
        }

        if (optimize_estimate_extrinsic_) {
            Pose ext;
            ext.t[0] = extrinsic_[0];
            ext.t[1] = extrinsic_[1];
            ext.t[2] = extrinsic_[2];

            Quaterniond qic = Quaterniond(extrinsic_[6], extrinsic_[3], extrinsic_[4], extrinsic_[5]);
            ext.R           = Rotation::quaternion2matrix(qic.normalized());

            // 外参估计检测, 误差较大则不更新, 1m or 5deg
            double dt = (ext.t - pose_b_c_.t).norm();
            double dr = Rotation::matrix2quaternion(ext.R * pose_b_c_.R.transpose()).vec().norm() * R2D;
            if ((dt > 1.0) || (dr > 5.0)) {
                LOGE << "Estimated extrinsic is too large, t: " << ext.t.transpose()
                     << ", R: " << Rotation::matrix2euler(ext.R).transpose() * R2D;
            } else {
                // Update the extrinsic
                Lock lock(extrinsic_mutex_);
                pose_b_c_ = ext;
            }

            vector<double> extrinsic;
            Vector3d euler = Rotation::matrix2euler(ext.R) * R2D;

            extrinsic.push_back(timelist_.back());
            extrinsic.push_back(ext.t[0]);
            extrinsic.push_back(ext.t[1]);
            extrinsic.push_back(ext.t[2]);
            extrinsic.push_back(euler[0]);
            extrinsic.push_back(euler[1]);
            extrinsic.push_back(euler[2]);
            extrinsic.push_back(td_b_c_);

            extfilesaver_->dump(extrinsic);
            extfilesaver_->flush();
        }
    }

    // 更新关键帧的位姿
    // Update the keyframe pose
    for (auto &keyframe : map_->keyframes()) {
        auto &frame = keyframe.second;
        auto index  = getStateDataIndex(frame->stamp());
        if (index < 0) {
            continue;
        }

        IntegrationState state = Preintegration::stateFromData(statedatalist_[index], preintegration_options_);
        frame->setPose(MISC::stateToCameraPose(state, pose_b_c_));
    }

    // 更新路标点的深度和位置
    // Update the mappoints
    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        auto frame = mappoint->referenceFrame();
        if (!frame || !map_->isKeyFrameInMap(frame)) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        double invdepth = invdepthlist_[mappoint->id()];
        double depth    = 1.0 / invdepth;

        auto pc0      = camera_->pixel2cam(mappoint->referenceKeypoint());
        Vector3d pc00 = {pc0.x(), pc0.y(), 1.0};
        pc00 *= depth;

        mappoint->pos() = camera_->cam2world(pc00, mappoint->referenceFrame()->pose());
        mappoint->updateDepth(depth);
    }

    // tmp change
    // {    
    //     auto [lock, sfm] = map_->SFMConstructMutable();
    //     for (auto &[id, feat]: sfm) {
    //         auto &observations = feat.obs;
    //         if (observations.size() < 4) continue;
    //         if (exinvdepthlist_.find(id) == exinvdepthlist_.end()) {
    //             continue;
    //         }

    //         double invdepth = exinvdepthlist_[id];
    //         double depth    = 1.0 / invdepth;

    //         feat.estimated_depth = depth;
    //     }
    // }
}

bool GVINS::gvinsRemoveAllSecondNewFrame() {
    vector<ulong> keyframeids = map_->orderedKeyFrames();

    for (auto id : keyframeids) {
        auto frame = map_->keyframes().find(id)->second;
        // 移除次新帧, 以及倒数第二个空关键帧
        if ((frame->keyFrameState() == KEYFRAME_REMOVE_SECOND_NEW) ||
            (frame->features().empty() && (id != keyframeids.back()))) {
            unused_time_nodes_.push_back(frame->stamp());

            // 仅需要重置关键帧标志, 从地图中移除次新关键帧即可,
            // 无需调整状态参数和路标点
            // Just remove the frame
            frame->resetKeyFrame();
            map_->removeKeyFrame(frame, false);
            // map_->moveKeyFrameToNonKeyFrame(frame);
        }
    }

    return true;
}

bool GVINS::gvinsIncrementalMarginalization() {
    std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数 -----------------------------------------------------------------------------------------------
    // For fixed order
    std::unordered_map<long, long> parameters_ids;
    parameters_ids.clear();
    long parameters_id = 0;

    // LOGI << "Last_marginalization_parameters_ids:";
    // 边缘化参数
    for (auto &last_marginalization_parameter_block : last_marginalization_parameter_blocks_) {
        parameters_ids[reinterpret_cast<long>(last_marginalization_parameter_block)] = parameters_id++;

        // long key = reinterpret_cast<long>(last_marginalization_parameter_block);
        // parameters_ids[key] = parameters_id;
        // LOGI << "[ParamID] key=0x" << std::hex << key
        //     << " -> id=" << std::dec << parameters_id;
        // ++parameters_id;
    }

    // 外参参数
    // Extrinsic parameters
    parameters_ids[reinterpret_cast<long>(extrinsic_)]     = parameters_id++;
    parameters_ids[reinterpret_cast<long>(extrinsic_ + 7)] = parameters_id++;

    // LOGI << "Extrinsic parameters_ids:";
    // {
    //     long key0 = reinterpret_cast<long>(extrinsic_);
    //     parameters_ids[key0] = parameters_id;
    //     LOGI << "[ParamID] key=0x" << std::hex << key0
    //         << " -> id=" << std::dec << parameters_id;
    //     ++parameters_id;

    //     long key1 = reinterpret_cast<long>(extrinsic_ + 7);
    //     parameters_ids[key1] = parameters_id;
    //     LOGI << "[ParamID] key=0x" << std::hex << key1
    //         << " -> id=" << std::dec << parameters_id;
    //     ++parameters_id;
    // }

    // LOGI << "Pose parameters_ids:";
    // 位姿参数
    // Pose parameters
    for (const auto &statedata : statedatalist_) {
        parameters_ids[reinterpret_cast<long>(statedata.pose)] = parameters_id++;
        parameters_ids[reinterpret_cast<long>(statedata.mix)]  = parameters_id++;

        // long key_pose = reinterpret_cast<long>(statedata.pose);
        // parameters_ids[key_pose] = parameters_id;
        // LOGI << "[ParamID] key=0x" << std::hex << key_pose
        //     << " -> id=" << std::dec << parameters_id;
        // ++parameters_id;

        // long key_mix = reinterpret_cast<long>(statedata.mix);
        // parameters_ids[key_mix] = parameters_id;
        // LOGI << "[ParamID] key=0x" << std::hex << key_mix
        //     << " -> id=" << std::dec << parameters_id;
        // ++parameters_id;
    }

    // LOGI << "Sub Pose parameters_ids:";
    // 位姿参数
    // Pose parameters
    for (size_t i = 1; i + 1 < sub_statedatalist_.size(); ++i) {
        const auto &statedata = sub_statedatalist_[i];
        parameters_ids[reinterpret_cast<long>(statedata.pose)] = parameters_id++;
        parameters_ids[reinterpret_cast<long>(statedata.mix)]  = parameters_id++;

        // long key_pose = reinterpret_cast<long>(statedata.pose);
        // parameters_ids[key_pose] = parameters_id;
        // LOGI << "[ParamID] key=0x" << std::hex << key_pose
        //     << " -> id=" << std::dec << parameters_id;
        // ++parameters_id;

        // long key_mix = reinterpret_cast<long>(statedata.mix);
        // parameters_ids[key_mix] = parameters_id;
        // LOGI << "[ParamID] key=0x" << std::hex << key_mix
        //     << " -> id=" << std::dec << parameters_id;
        // ++parameters_id;
    }

    // LOGI << "Inverse depth parameters_ids:";
    // 逆深度参数
    // Inverse depth parameters
    vector<ulong> keyframeids = map_->orderedKeyFrames();
    auto ref_frame = map_->keyframes().find(keyframeids[keyframeids.size() - 2])->second;
    auto features = ref_frame->features();

    for (auto const &[mappointid, feature] : features) {
        auto mappoint = feature->getMapPoint();
        if (feature->isOutlier() || !mappoint || mappoint->isOutlier()) {
            continue;
        }

        //tmp change
        // if (mappoint->observations().size() < 4) {
        //     continue;
        // }

        if (sub_invdepthlist_.find(mappointid) == sub_invdepthlist_.end()) {
            continue;
        }

        double *invdepth                                 = &sub_invdepthlist_[mappointid];
        parameters_ids[reinterpret_cast<long>(invdepth)] = parameters_id++;
        // long key = reinterpret_cast<long>(invdepth);
        // parameters_ids[key] = parameters_id;
        // LOGI << "[ParamID] key=0x" << std::hex << key
        //     << " -> id=" << std::dec << parameters_id;
        // ++parameters_id;
    }

    // 更新参数块的特定ID, 必要的
    // Update the IS for parameters
    marginalization_info->updateParamtersIds(parameters_ids);

    // 添加残差块 --------------------------------------------------------------------------------------------------------------------

    // 边缘化因子
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {

        std::vector<int> marginalized_index;
        for (size_t i = 1; i + 1 < sub_statedatalist_.size(); ++i) {
            const auto &statedata = sub_statedatalist_[i];
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); ++k) {
                if (last_marginalization_parameter_blocks_[k] == statedata.pose ||
                    last_marginalization_parameter_blocks_[k] == statedata.mix) {
                    marginalized_index.push_back((int) k);
                }
            }
        }

        auto factor   = std::make_shared<MarginalizationFactor>(last_marginalization_info_);
        auto residual = std::make_shared<ResidualBlockInfo>(factor, nullptr, last_marginalization_parameter_blocks_,
                                                            marginalized_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 预积分因子
    // The IMU preintegration factors
    size_t num_marg = sub_statedatalist_.size();
    for (size_t k = 0; k < num_marg - 1; k++) {
        // 由于会移除多个预积分, 会导致出现保留和移除同时出现, 判断索引以区分
        // More than one may be removed
        vector<int> marg_index;
        auto factor   = std::make_shared<PreintegrationFactor>(sub_preintegrationlist_[k]);
        std::shared_ptr<ResidualBlockInfo> residual;
        if (k == (num_marg - 2)) {
            marg_index = {0, 1};
            residual = std::make_shared<ResidualBlockInfo>(
            factor, nullptr,
            std::vector<double *>{sub_statedatalist_[k].pose, sub_statedatalist_[k].mix, statedatalist_.back().pose,
                                  statedatalist_.back().mix}, marg_index);
        } else if (k == 0) {
            marg_index = {2, 3};
            residual = std::make_shared<ResidualBlockInfo>(
            factor, nullptr,
            std::vector<double *>{statedatalist_[statedatalist_.size() - 2].pose, statedatalist_[statedatalist_.size() - 2].mix, sub_statedatalist_[k + 1].pose,
                                  sub_statedatalist_[k + 1].mix}, marg_index);
        } else {
            marg_index = {0, 1, 2, 3};
            residual = std::make_shared<ResidualBlockInfo>(
            factor, nullptr,
            std::vector<double *>{sub_statedatalist_[k].pose, sub_statedatalist_[k].mix, sub_statedatalist_[k + 1].pose,
                                  sub_statedatalist_[k + 1].mix}, marg_index);
        }

        marginalization_info->addResidualBlockInfo(residual);
    }

    // 重投影因子, 最老的关键帧
    // The visual reprojection factors
    auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
    for (auto const &[mappointid, ref_feature] : features) {
        auto mappoint = ref_feature->getMapPoint();
        if (ref_feature->isOutlier() || !mappoint || mappoint->isOutlier()) {
            continue;
        }

        auto ref_frame_pc = ref_feature->point();

        if (sub_invdepthlist_.find(mappointid) == sub_invdepthlist_.end()) {
            continue;
        }

        double *invdepth = &sub_invdepthlist_[mappointid];

        auto observations = mappoint->observations();
        //tmp change
        // if (observations.size() < 4) continue;
        for (auto &observation : observations) {
            auto obs_feature = observation.second.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            if (!obs_frame || !map_->isSubKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            auto obs_frame_pc = camera_->pixel2cam(obs_feature->keyPoint());
            size_t obs_frame_index;
            auto it = std::find(sub_timelist_.begin(), sub_timelist_.end(), obs_frame->stamp());
            if (it != sub_timelist_.end()) {
                obs_frame_index = std::distance(sub_timelist_.begin(), it);
            } else {
                LOGE << "Wrong matched mapoint keyframes ";
                continue;
            }

            auto factor = std::make_shared<ReprojectionFactor>(
                ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(), obs_feature->velocityInPixel(),
                ref_frame->timeDelay(), obs_frame->timeDelay(), optimize_reprojection_error_std_);

            vector<int> marg_index;
            std::shared_ptr<ResidualBlockInfo> residual;
            if (obs_frame_index == (sub_timelist_.size() - 1)) {
                marg_index = {3};
                residual = std::make_shared<ResidualBlockInfo>(factor, loss_function,
                                                    vector<double *>{statedatalist_[statedatalist_.size() - 2].pose,
                                                                        statedatalist_.back().pose,
                                                                        extrinsic_, invdepth, &extrinsic_[7]},
                                                    marg_index);
            } else {
                marg_index = {1, 3};
                residual = std::make_shared<ResidualBlockInfo>(factor, loss_function,
                                                    vector<double *>{statedatalist_[statedatalist_.size() - 2].pose,
                                                                        sub_statedatalist_[obs_frame_index].pose,
                                                                        extrinsic_, invdepth, &extrinsic_[7]},
                                                    marg_index);
            }

            marginalization_info->addResidualBlockInfo(residual);
        }
    }

    // 边缘化处理 ----------------------------------------------------------------------------------------------------------------------
    // Do marginalization
    marginalization_info->marginalization();

    // 保留的数据, 使用独立ID -----------------------------------------------------------------------------------------------------------
    // Update the address
    std::unordered_map<long, double *> address;
    for (size_t k = 0; k < statedatalist_.size(); k++) {
        address[parameters_ids[reinterpret_cast<long>(statedatalist_[k].pose)]] = statedatalist_[k].pose;
        address[parameters_ids[reinterpret_cast<long>(statedatalist_[k].mix)]]  = statedatalist_[k].mix;
    }
    address[parameters_ids[reinterpret_cast<long>(extrinsic_)]] = extrinsic_;
    address[parameters_ids[reinterpret_cast<long>(extrinsic_ + 7)]] = &extrinsic_[7];

    last_marginalization_parameter_blocks_ = marginalization_info->getParamterBlocks(address);
    last_marginalization_info_             = std::move(marginalization_info);

    return true;
}

bool GVINS::gvinsMarginalization() {
    // 按时间先后排序的关键帧
    // Ordered keyframes
    vector<ulong> keyframeids = map_->orderedKeyFrames();
    auto latest_keyframe      = map_->latestKeyFrame();

    latest_keyframe->setKeyFrameState(KEYFRAME_NORMAL);

    // 对齐到保留的最后一个关键帧, 可能移除多个预积分对象
    // Align to the last keyframe time
    auto frame      = map_->keyframes().find(keyframeids[1])->second;
    int num_marg = getStateDataIndex(frame->stamp());

    double last_time = timelist_[num_marg];

    LOGI << "Marginalize " << num_marg << " states, last time " << Logging::doubleData(last_time);

    std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数
    // For fixed order
    std::unordered_map<long, long> parameters_ids;
    parameters_ids.clear();
    long parameters_id = 0;

    {
        // 边缘化参数
        // Marginalization parameters
        for (auto &last_marginalization_parameter_block : last_marginalization_parameter_blocks_) {
            parameters_ids[reinterpret_cast<long>(last_marginalization_parameter_block)] = parameters_id++;
        }

        // 外参参数
        // Extrinsic parameters
        parameters_ids[reinterpret_cast<long>(extrinsic_)]     = parameters_id++;
        parameters_ids[reinterpret_cast<long>(extrinsic_ + 7)] = parameters_id++;

        // 位姿参数
        // Pose parameters
        for (const auto &statedata : statedatalist_) {
            parameters_ids[reinterpret_cast<long>(statedata.pose)] = parameters_id++;
            parameters_ids[reinterpret_cast<long>(statedata.mix)]  = parameters_id++;
        }

        // 逆深度参数
        // Inverse depth parameters
        frame         = map_->keyframes().at(keyframeids[0]);
        auto features = frame->features();
        for (auto const &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
                continue;
            }

            if (mappoint->referenceFrame() != frame) {
                continue;
            }

            //tmp change
            if (mappoint->observations().size() < 4) {
                continue;
            }

            double *invdepth                                 = &invdepthlist_[mappoint->id()];
            parameters_ids[reinterpret_cast<long>(invdepth)] = parameters_id++;
        }

        // 更新参数块的特定ID, 必要的
        // Update the IS for parameters
        marginalization_info->updateParamtersIds(parameters_ids);
    }

    // 边缘化因子
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {

        std::vector<int> marginalized_index;
        for (size_t i = 0; i < num_marg; i++) {
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++) {
                if (last_marginalization_parameter_blocks_[k] == statedatalist_[i].pose ||
                    last_marginalization_parameter_blocks_[k] == statedatalist_[i].mix) {
                    marginalized_index.push_back((int) k);
                }
            }
        }

        auto factor   = std::make_shared<MarginalizationFactor>(last_marginalization_info_);
        auto residual = std::make_shared<ResidualBlockInfo>(factor, nullptr, last_marginalization_parameter_blocks_,
                                                            marginalized_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 预积分因子
    // The IMU preintegration factors
    for (size_t k = 0; k < num_marg; k++) {
        // 由于会移除多个预积分, 会导致出现保留和移除同时出现, 判断索引以区分
        // More than one may be removed
        vector<int> marg_index;
        if (k == (num_marg - 1)) {
            marg_index = {0, 1};
        } else {
            marg_index = {0, 1, 2, 3};
        }

        auto factor   = std::make_shared<PreintegrationFactor>(preintegrationlist_[k]);
        auto residual = std::make_shared<ResidualBlockInfo>(
            factor, nullptr,
            std::vector<double *>{statedatalist_[k].pose, statedatalist_[k].mix, statedatalist_[k + 1].pose,
                                  statedatalist_[k + 1].mix},
            marg_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 先验约束因子
    // The prior factor
    if (is_use_prior_) {
        auto pose_factor   = std::make_shared<ImuPosePriorFactor>(pose_prior_, pose_prior_std_);
        auto pose_residual = std::make_shared<ResidualBlockInfo>(
            pose_factor, nullptr, std::vector<double *>{statedatalist_[0].pose}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(pose_residual);

        auto mix_factor   = std::make_shared<ImuMixPriorFactor>(preintegration_options_, mix_prior_, mix_prior_std_);
        auto mix_residual = std::make_shared<ResidualBlockInfo>(
            mix_factor, nullptr, std::vector<double *>{statedatalist_[0].mix}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(mix_residual);

        is_use_prior_ = false;
    }

    // 重投影因子, 最老的关键帧
    // The visual reprojection factors

    frame         = map_->keyframes().at(keyframeids[0]);
    auto features = frame->features();

    auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
    for (auto const &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
            continue;
        }

        auto ref_frame = mappoint->referenceFrame();
        if (ref_frame != frame) {
            continue;
        }

        auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
        int ref_frame_index = getStateDataIndex(ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }

        double *invdepth = &invdepthlist_[mappoint->id()];

        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        auto observations = mappoint->observations();
        //tmp change
        if (observations.size() < 4) continue;
        for (auto &observation : observations) {
            auto obs_feature = observation.second.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            if (!obs_frame || !obs_frame->isKeyFrame() || !map_->isKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
            int obs_frame_index = getStateDataIndex(obs_frame->stamp());

            if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
                LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
                     << Logging::doubleData(obs_frame->stamp());
                continue;
            }

            auto factor = std::make_shared<ReprojectionFactor>(
                ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(), obs_feature->velocityInPixel(),
                ref_frame->timeDelay(), obs_frame->timeDelay(), optimize_reprojection_error_std_);
            auto residual = std::make_shared<ResidualBlockInfo>(factor, loss_function,
                                                                vector<double *>{statedatalist_[ref_frame_index].pose,
                                                                                 statedatalist_[obs_frame_index].pose,
                                                                                 extrinsic_, invdepth, &extrinsic_[7]},
                                                                vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual);
        }
    }

    // 边缘化处理
    // Do marginalization
    marginalization_info->marginalization();

    // 保留的数据, 使用独立ID
    // Update the address
    std::unordered_map<long, double *> address;
    for (size_t k = num_marg; k < statedatalist_.size(); k++) {
        address[parameters_ids[reinterpret_cast<long>(statedatalist_[k].pose)]] = statedatalist_[k].pose;
        address[parameters_ids[reinterpret_cast<long>(statedatalist_[k].mix)]]  = statedatalist_[k].mix;
    }
    address[parameters_ids[reinterpret_cast<long>(extrinsic_)]]     = extrinsic_;
    address[parameters_ids[reinterpret_cast<long>(extrinsic_ + 7)]] = &extrinsic_[7];

    last_marginalization_parameter_blocks_ = marginalization_info->getParamterBlocks(address);
    last_marginalization_info_             = std::move(marginalization_info);

    // 移除边缘化的数据
    // Remove the marginalized data

    // 预积分观测及时间状态
    // The IMU preintegration and time nodes
    for (size_t k = 0; k < num_marg; k++) {
        timelist_.pop_front();
        statedatalist_.pop_front();
        preintegrationlist_.pop_front();
    }

    // 保存移除的路标点, 用于可视化
    // The marginalized mappoints, for visualization
    frame    = map_->keyframes().at(keyframeids[0]);
    features = frame->features();
    for (const auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
            continue;
        }
        auto &pw = mappoint->pos();

        if (is_use_visualization_) {
            drawer_->addNewFixedMappoint(pw);
        }

        // 保存路标点
        // Save these mappoints to file
        ptsfilesaver_->dump(vector<double>{pw.x(), pw.y(), pw.z()});
    }

    // 关键帧
    // The marginalized keyframe
    map_->removeKeyFrame(frame, true);

    return true;
}

void GVINS::doReintegration() {
    int cnt = 0;
    for (size_t k = 0; k < preintegrationlist_.size(); k++) {
        IntegrationState state = Preintegration::stateFromData(statedatalist_[k], preintegration_options_);
        Vector3d dbg           = preintegrationlist_[k]->deltaState().bg - state.bg;
        Vector3d dba           = preintegrationlist_[k]->deltaState().ba - state.ba;
        if ((dbg.norm() > 6 * integration_parameters_->gyr_bias_std) ||
            (dba.norm() > 6 * integration_parameters_->acc_bias_std)) {//6 6
            preintegrationlist_[k]->reintegration(state);
            cnt++;
        }
    }
    if (cnt) {
        LOGW << "Reintegration " << cnt << " preintegration";
    }
}

void GVINS::addReprojectionParameters(ceres::Problem &problem) {
    if (map_->landmarks().empty()) {
        return;
    }

    invdepthlist_.clear();
    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            auto frame = mappoint->referenceFrame();
            if (!frame || !map_->isKeyFrameInMap(frame)) {
                continue;
            }

            double depth         = mappoint->depth();
            double inverse_depth = 1.0 / depth;

            // 确保深度数值有效
            // For valid mappoints
            if (std::isnan(inverse_depth)) {
                mappoint->setOutlier(true);
                LOGE << "Mappoint " << mappoint->id() << " is wrong with depth " << depth << " type "
                     << mappoint->mapPointType();
                continue;
            }

            invdepthlist_[mappoint->id()] = inverse_depth;
            problem.AddParameterBlock(&invdepthlist_[mappoint->id()], 1);

            mappoint->addOptimizedTimes();
        }
    }

    // // tmp change
    // {            
    //     exinvdepthlist_.clear();
    //     auto [lock, sfm] = map_->SFMConstructMutable();
    //     for (auto &[id, feat]: sfm) {
    //         auto &observations = feat.obs;
    //         if (observations.size() < 4) continue;
    //         auto obs_it = observations.begin();
    //         while (obs_it != observations.end() && obs_it->first < (timelist_[0] - 0.001)) {
    //             obs_it = observations.erase(obs_it);
    //         }
    //         if (observations.size() < 4) continue;
    //         double depth = feat.estimated_depth;
    //         if (depth <= 0) {
    //             depth = 5.0;
    //         }
    //         double inverse_depth = 1.0 / depth;
    //         exinvdepthlist_[id] = inverse_depth;
    //         problem.AddParameterBlock(&exinvdepthlist_[id], 1);
    //     }
    // }

    // 外参
    // Extrinsic parameters
    extrinsic_[0] = pose_b_c_.t[0];
    extrinsic_[1] = pose_b_c_.t[1];
    extrinsic_[2] = pose_b_c_.t[2];

    Quaterniond qic = Rotation::matrix2quaternion(pose_b_c_.R);
    qic.normalize();
    extrinsic_[3] = qic.x();
    extrinsic_[4] = qic.y();
    extrinsic_[5] = qic.z();
    extrinsic_[6] = qic.w();

    ceres::LocalParameterization *parameterization = new (PoseParameterization);
    problem.AddParameterBlock(extrinsic_, 7, parameterization);

    if (!optimize_estimate_extrinsic_ || gvinsstate_ != GVINS_TRACKING_NORMAL) {
        problem.SetParameterBlockConstant(extrinsic_);
    }

    // 时间延时
    // Time delay
    extrinsic_[7] = td_b_c_;
    problem.AddParameterBlock(&extrinsic_[7], 1);
    if (!optimize_estimate_td_ || gvinsstate_ != GVINS_TRACKING_NORMAL) {
        problem.SetParameterBlockConstant(&extrinsic_[7]);
    }
}

vector<ceres::ResidualBlockId> GVINS::addReprojectionFactors(ceres::Problem &problem, bool isusekernel) {

    vector<ceres::ResidualBlockId> residual_ids;

    if (map_->keyframes().empty()) {
        return residual_ids;
    }

    ceres::LossFunction *loss_function = nullptr;
    if (isusekernel) {
        loss_function = new ceres::HuberLoss(1.0);
    }

    residual_ids.clear();
    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        auto ref_frame = mappoint->referenceFrame();
        if (!map_->isKeyFrameInMap(ref_frame)) {
            continue;
        }

        auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
        int ref_frame_index = getStateDataIndex(ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }

        double *invdepth = &invdepthlist_[mappoint->id()];
        if (*invdepth == 0) {
            *invdepth = 1.0 / MapPoint::DEFAULT_DEPTH;
        }

        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        auto observations = mappoint->observations();
        for (auto &observation : observations) {
            auto obs_feature = observation.second.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            if (!obs_frame || !obs_frame->isKeyFrame() || !map_->isKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
            int obs_frame_index = getStateDataIndex(obs_frame->stamp());

            if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
                LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
                     << Logging::doubleData(obs_frame->stamp());
                continue;
            }

            auto factor = new ReprojectionFactor(ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(),
                                                 obs_feature->velocityInPixel(), ref_frame->timeDelay(),
                                                 obs_frame->timeDelay(), optimize_reprojection_error_std_);
            auto residual_block_id =
                problem.AddResidualBlock(factor, loss_function, statedatalist_[ref_frame_index].pose,
                                         statedatalist_[obs_frame_index].pose, extrinsic_, invdepth, &extrinsic_[7]);
            residual_ids.push_back(residual_block_id);
        }
    }

    return residual_ids;
}

int GVINS::getStateDataIndex(double time) {

    size_t index = MISC::getStateDataIndex(timelist_, time, MISC::MINIMUM_TIME_INTERVAL);
    if (!MISC::isTheSameTimeNode(timelist_[index], time, MISC::MINIMUM_TIME_INTERVAL)) {
        LOGE << "Wrong matching time node " << Logging::doubleData(timelist_[index]) << " to "
             << Logging::doubleData(time);
        return -1;
    }
    return static_cast<int>(index);
}

void GVINS::addStateParameters(ceres::Problem &problem) {
    // LOGI << "Total " << statedatalist_.size() << " pose states from "
    //      << Logging::doubleData(statedatalist_.begin()->time) << " to "
    //      << Logging::doubleData(statedatalist_.back().time);

    // ==== 新增：打印所有参与优化的帧的时间戳 ====
    // LOGI << "Participating state timestamps:";
    // for (const auto &statedata : statedatalist_) {
    //     LOGI << std::fixed << std::setprecision(9) << "  - time: " << statedata.time;
    // }

    for (auto &statedata : statedatalist_) {
        // 位姿
        // Pose
        ceres::LocalParameterization *parameterization = new (PoseParameterization);
        problem.AddParameterBlock(statedata.pose, Preintegration::numPoseParameter(), parameterization);

        // IMU mix parameters
        problem.AddParameterBlock(statedata.mix, Preintegration::numMixParameter(preintegration_options_));

    }

    // ==== 打印每个状态的时间和位姿 ====
    // double *pose = statedatalist_.back().pose;
    // Eigen::Quaterniond q(pose[6], pose[3], pose[4], pose[5]); // (w, x, y, z)
    // Eigen::Vector3d t(pose[0], pose[1], pose[2]);

    // LOGI << "The Newest Time: " << std::fixed << std::setprecision(9) << statedatalist_.back().time
    //         << " | Translation: [" << t.transpose() << "]"
    //         << " | Quaternion: [" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "]";
}

void GVINS::addImuFactors(ceres::Problem &problem) {
    for (size_t k = 0; k < preintegrationlist_.size(); k++) {
        // 预积分因子
        // IMU preintegration factors
        auto factor = new PreintegrationFactor(preintegrationlist_[k]);
        problem.AddResidualBlock(factor, nullptr, statedatalist_[k].pose, statedatalist_[k].mix,
                                 statedatalist_[k + 1].pose, statedatalist_[k + 1].mix);
    }

    // 添加IMU误差约束, 限制过大的误差估计
    // IMU error factor
    // auto factor = new ImuErrorFactor(preintegration_options_);
    // problem.AddResidualBlock(factor, nullptr, statedatalist_[preintegrationlist_.size()].mix);

    // IMU初始先验因子, 仅限于初始化
    // IMU prior factor, only for initialization
    if (is_use_prior_) {
        auto pose_factor = new ImuPosePriorFactor(pose_prior_, pose_prior_std_);
        problem.AddResidualBlock(pose_factor, nullptr, statedatalist_[0].pose);

        auto mix_factor = new ImuMixPriorFactor(preintegration_options_, mix_prior_, mix_prior_std_);
        problem.AddResidualBlock(mix_factor, nullptr, statedatalist_[0].mix);
    }
}

void GVINS::addImuFactors_loss(ceres::Problem &problem) {
    auto* loss_function = new ceres::CauchyLoss(1.0);  // 1.0 为鲁棒因子的尺度，可调 //nullptr

    for (size_t k = 0; k < preintegrationlist_.size(); k++) {
        // 预积分因子
        // IMU preintegration factors
        auto factor = new PreintegrationFactor(preintegrationlist_[k]);
        problem.AddResidualBlock(factor, loss_function, statedatalist_[k].pose, statedatalist_[k].mix,
                                 statedatalist_[k + 1].pose, statedatalist_[k + 1].mix);
    }

    // 添加IMU误差约束, 限制过大的误差估计
    // IMU error factor
    auto factor = new ImuErrorFactor(preintegration_options_);
    problem.AddResidualBlock(factor, nullptr, statedatalist_[preintegrationlist_.size()].mix);

    // IMU初始先验因子, 仅限于初始化
    // IMU prior factor, only for initialization
    if (is_use_prior_) {
        auto pose_factor = new ImuPosePriorFactor(pose_prior_, pose_prior_std_);
        problem.AddResidualBlock(pose_factor, nullptr, statedatalist_[0].pose);

        auto mix_factor = new ImuMixPriorFactor(preintegration_options_, mix_prior_, mix_prior_std_);
        problem.AddResidualBlock(mix_factor, nullptr, statedatalist_[0].mix);
    }
}

void GVINS::constructPrior(bool is_zero_velocity) {
    double pos_prior_std  = 0.1;                                       // 0.1 m
    double att_prior_std  = 0.5 * D2R;                                 // 0.5 deg
    double vel_prior_std  = 0.1;                                       // 0.1 m/s
    double bg_prior_std   = integration_parameters_->gyr_bias_std * 3; // Bias std * 3
    double ba_prior_std   = ACCELEROMETER_BIAS_PRIOR_STD;              // 20000 mGal
    double sodo_prior_std = 0.005;                                     // 5000 PPM

    if (!is_zero_velocity) {
        bg_prior_std = GYROSCOPE_BIAS_PRIOR_STD; // 7200 deg/hr
    }

    memcpy(pose_prior_, statedatalist_[0].pose, sizeof(double) * 7);
    memcpy(mix_prior_, statedatalist_[0].mix, sizeof(double) * 18);
    for (size_t k = 0; k < 3; k++) {
        pose_prior_std_[k + 0] = pos_prior_std;
        pose_prior_std_[k + 3] = att_prior_std;

        mix_prior_std_[k + 0] = vel_prior_std;
        mix_prior_std_[k + 3] = bg_prior_std;
        mix_prior_std_[k + 6] = ba_prior_std;
    }
    pose_prior_std_[5] = att_prior_std * 3; // heading
    mix_prior_std_[9]  = sodo_prior_std;
    is_use_prior_      = true;
}

void GVINS::slidewindow()
{
    // 按时间先后排序的关键帧
    // Ordered keyframes
    vector<ulong> keyframeids = map_->orderedKeyFrames();
    auto latest_keyframe      = map_->latestKeyFrame();

    // 对齐到保留的最后一个关键帧, 可能移除多个预积分对象
    // Align to the last keyframe time
    auto frame      = map_->keyframes().find(keyframeids[1])->second;
    size_t num_marg = getStateDataIndex(frame->stamp());

    double last_time = timelist_[num_marg];

    LOGI << "Slide window: " << num_marg << " states, last time: " << Logging::doubleData(last_time);

    // 移除边缘化的数据
    // Remove the marginalized data

    // 预积分观测及时间状态
    // The IMU preintegration and time nodes
    for (size_t k = 0; k < num_marg; k++) {
        timelist_.pop_front();
        statedatalist_.pop_front();
        preintegrationlist_.pop_front();
    }

    // 保存移除的路标点, 用于可视化
    // The marginalized mappoints, for visualization
    frame    = map_->keyframes().at(keyframeids[0]);
    auto features = frame->features();
    for (const auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
            continue;
        }
        auto &pw = mappoint->pos();

        if (is_use_visualization_) {
            drawer_->addNewFixedMappoint(pw);
        }

        // 保存路标点
        // Save these mappoints to file
        ptsfilesaver_->dump(vector<double>{pw.x(), pw.y(), pw.z()});
    }

    // 关键帧
    // The marginalized keyframe
    map_->removeKeyFrame(frame, true);
}

vector<ResidualInfo> GVINS::addReprojectionFactors_old(ceres::Problem &problem, bool isusekernel) {
    vector<ResidualInfo> residual_info_ids;

    if (map_->keyframes().empty()) {
        return residual_info_ids;
    }

    ceres::LossFunction *loss_function = nullptr;
    if (isusekernel) {
        loss_function = new ceres::HuberLoss(1.0);
    }

    residual_info_ids.clear();
    // std::unordered_map<uint64_t, std::vector<std::pair<size_t, double>>> map_point_rpe_map;
    // for (const auto &landmark : map_->landmarks()) {
    //     const auto &mappoint = landmark.second;
    //     if (!mappoint || mappoint->isOutlier()) {
    //         continue;
    //     }
    //     if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
    //         continue;
    //     }
    //     auto ref_frame = mappoint->referenceFrame();
    //     if (!map_->isKeyFrameInMap(ref_frame)) {
    //         continue;
    //     }
    //     auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
    //     size_t ref_frame_index = getStateDataIndex(ref_frame->stamp());
    //     if (ref_frame_index < 0) {
    //         continue;
    //     }
    //     double *invdepth = &invdepthlist_[mappoint->id()];
    //     if (*invdepth == 0) {
    //         *invdepth = 1.0 / MapPoint::DEFAULT_DEPTH;
    //     }
    //     auto ref_feature = ref_frame->features().find(mappoint->id())->second;
    //     auto observations = mappoint->observations();
    //     auto pw = mappoint->pos();
    //     for (auto &observation : observations) {
    //         auto obs_feature = observation.second.lock();
    //         if (!obs_feature || obs_feature->isOutlier()) {
    //             continue;
    //         }
    //         auto obs_frame = obs_feature->getFrame();
    //         if (!obs_frame || !obs_frame->isKeyFrame() || !map_->isKeyFrameInMap(obs_frame) ||
    //             (obs_frame == ref_frame)) {
    //             continue;
    //         }
    //         auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
    //         size_t obs_frame_index = getStateDataIndex(obs_frame->stamp());
    //         if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
    //             LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
    //                  << Logging::doubleData(obs_frame->stamp());
    //             continue;
    //         }
    //         auto obs_pose = obs_frame->pose();
    //         auto obs_err = camera_->reprojectionError(obs_pose, pw, obs_feature->keyPoint()).norm();
    //         map_point_rpe_map[mappoint->id()].emplace_back(obs_frame_index, obs_err);
    //     }
    // }
    // // 2) 统计均值、方差、计数，并计算权重
    // std::unordered_map<uint64_t, RpeStats> stats_map;
    // for (auto &kv : map_point_rpe_map) {
    //     uint64_t id = kv.first;
    //     auto &vec = kv.second;
    //     int N = int(vec.size());
    //     if (N == 0) continue;
    //     double sum = 0;
    //     for (auto &p : vec) sum += p.second;
    //     double mean = sum / N;
    //     double sq = 0;
    //     for (auto &p : vec) {
    //         double d = p.second - mean;
    //         sq += d*d;
    //     }
    //     double var = sq / N;
    //     stats_map[id] = { mean, var, N };
    // }
    // // 3) 追加写入文件
    // std::unordered_map<uint64_t, double > weight_map;
    // std::ofstream ofs("/sad/catkin_ws/ex_logs/map_point_rpe.txt", std::ios::app);
    // if (!ofs.is_open()) {
    //     std::cerr << "Failed to open output file for MapPoint RPEs!" << std::endl;
    // }
    // // 写一行分段标题
    // ofs << "[MapPointID | count | mean | var | weight | (frame,rpe)...]\n";
    // for (auto &kv : map_point_rpe_map) {
    //     uint64_t id = kv.first;
    //     auto &vec = kv.second;
    //     auto &st = stats_map[id];
    //     // 计算权重
    //     double w_mu    = std::exp(- (st.mean*st.mean) / (2*alpha_val*alpha_val));
    //     double w_var   = std::exp(- (st.var)         / (2*beta_val*beta_val));
    //     double w_count = st.count / (st.count + gamma_val);
    //     double w       = w_mu * w_var * w_count;
    //     weight_map[id] = 0.85 * w + 0.15 * 1.0;//0.8 * w + 0.2 * 1.1//0.85 * w + 0.15 * 1.0
    //     // 输出 ID, count, mean, var, weight
    //     ofs << std::left << std::setw(12) << id
    //         << std::right << std::setw(6) << st.count
    //         << std::fixed << std::setprecision(4)
    //         << std::setw(10) << st.mean
    //         << std::setw(10) << st.var
    //         << std::setw(10) << w;
    //     // 输出每个观测 (frame_idx,rpe)
    //     for (auto &p : vec) {
    //         ofs << std::setw(6) << p.first
    //             << std::setw(10) << p.second;
    //     }
    //     ofs << "\n";
    // }
    // // 分段横线
    // ofs << std::string(100,'-') << "\n";
    // ofs.close();

    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        auto ref_frame = mappoint->referenceFrame();
        if (!map_->isKeyFrameInMap(ref_frame)) {
            continue;
        }

        auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
        int ref_frame_index = getStateDataIndex(ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }

        auto observations = mappoint->observations();
        //tmp change
        double w = 1.0;
        if (observations.size() < 4) {
            w = 0.0;
            continue;
        } else {
            w = 1.0;
        }
        // double w = 1.0;
        // auto itw = weight_map.find(mappoint->id());
        // if (itw != weight_map.end()) {
        //     w = itw->second;
        // }

        double *invdepth = &invdepthlist_[mappoint->id()];
        if (*invdepth == 0) {
            *invdepth = 1.0 / MapPoint::DEFAULT_DEPTH;
        }

        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        for (auto &observation : observations) {
            auto obs_feature = observation.second.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            if (!obs_frame || !obs_frame->isKeyFrame() || !map_->isKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
            int obs_frame_index = getStateDataIndex(obs_frame->stamp());

            if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
                LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
                     << Logging::doubleData(obs_frame->stamp());
                continue;
            }

            auto factor = new ReprojectionFactorW(ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(),
                                                 obs_feature->velocityInPixel(), ref_frame->timeDelay(),
                                                 obs_frame->timeDelay(), optimize_reprojection_error_std_);
            factor->set_weight(w);
            
            auto residual_block_id =
                problem.AddResidualBlock(factor, loss_function, statedatalist_[ref_frame_index].pose,
                                         statedatalist_[obs_frame_index].pose, extrinsic_, invdepth, &extrinsic_[7]);

            residual_info_ids.emplace_back(w, residual_block_id, factor);
        }
    }

    // // tmp change
    // {    
    //     auto [lock, sfm] = map_->SFMConstruct();
    //     for (auto &[id, feat]: sfm) {
    //         auto &observations = feat.obs;
    //         if (observations.size() < 4) continue;
    //         if (exinvdepthlist_.find(id) == exinvdepthlist_.end()) {
    //             continue;
    //         }
    //         // LOGI << "Feature ID: " << id << ", obs.size(): " << observations.size();
    //         auto obs_it = observations.begin();
    //         double ref_time = obs_it->first;
    //         // LOGI << "Ref time: " << std::fixed << std::setprecision(9) << ref_time;
    //         while (obs_it != observations.end() && std::find(timelist_.begin(), timelist_.end(), ref_time) == timelist_.end()) {
    //             ++obs_it;
    //             ref_time = obs_it->first;
    //         }
    //         if (obs_it == observations.end()) continue;
    //         double ref_td = map_->getframebytime(ref_time)->timeDelay();
    //         auto ref_frame_pc = obs_it->second.point();
    //         auto ref_frame_vel = obs_it->second.velocity3d();
    //         int ref_frame_index = getStateDataIndex(ref_time);
    //         if (ref_frame_index < 0) {
    //             continue;
    //         }
    //         double *invdepth = &exinvdepthlist_[id];
    //         if (*invdepth == 0) {
    //             *invdepth = 1.0 / MapPoint::DEFAULT_DEPTH;
    //         }
    //         ++obs_it;
    //         double w = 0.0;
    //         while(obs_it != observations.end()) {
    //             double obs_time = obs_it->first;
    //             auto obs_frame = map_->getframebytime(obs_time);
    //             if (!obs_frame) {
    //                 ++obs_it;
    //                 continue;
    //             }
    //             double obs_td = obs_frame->timeDelay();
    //             auto obs_frame_pc = obs_it->second.point();
    //             auto obs_frame_vel = obs_it->second.velocity3d();
    //             int obs_frame_index = getStateDataIndex(obs_time);
    //             if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
    //                 ++obs_it;
    //                 continue;
    //             }
    //             // LOG(INFO) << "Obs time: " << std::fixed << std::setprecision(9) << obs_time
    //             //     << ", obs_frame_index: " << obs_frame_index
    //             //     << ", ref_frame_index: " << ref_frame_index;
    //             auto factor = new ReprojectionFactorW(ref_frame_pc, obs_frame_pc, ref_frame_vel,
    //                                                 obs_frame_vel, ref_td, obs_td, optimize_reprojection_error_std_);
    //             factor->set_weight(w); 
    //             auto residual_block_id =
    //                 problem.AddResidualBlock(factor, loss_function, statedatalist_[ref_frame_index].pose,
    //                                         statedatalist_[obs_frame_index].pose, extrinsic_, invdepth, &extrinsic_[7]);
    //             residual_info_ids.emplace_back(w, residual_block_id, factor);
    //             ++obs_it;
    //         }
    //     }
    // }

    return residual_info_ids;
}

int GVINS::removeReprojectionFactorsByChi2_old(ceres::Problem &problem, vector<ResidualInfo> &residual_ids,
                                           double chi2) {
    double cost;
    int outlier_features = 0;
    double average_cost = 0;

    // 进行卡方检验, 判定粗差因子, 待全部判定完成再进行移除, 否则会导致错误
    // Judge first and remove later
    vector<ceres::ResidualBlockId> outlier_residual_ids;
    for (auto &info : residual_ids) {
        auto &id = info.id;
        double w = info.w;
        if (w == 0) continue;
        problem.EvaluateResidualBlock(id, false, &cost, nullptr, nullptr);
        cost = 2 * cost / (w * w);
        average_cost += cost;

        // cost带有1/2系数
        // To chi2
        if (cost > chi2) {
            outlier_features++;
            outlier_residual_ids.push_back(id);
        }
    }

    // 从优化问题中移除所有粗差因子
    // Remove the outliers from the optimizer
    for (auto &id : outlier_residual_ids) {
        problem.RemoveResidualBlock(id);
    }

    average_cost /= residual_ids.size();

    LOGI << "Remove " << outlier_features << " reprojection factors with average cost " << average_cost;

    return outlier_features;
}

bool GVINS::gvinsOptimization_old() {
    static int first_num_iterations  = 15;
    static int second_num_iterations = 8;

    TimeCost timecost;

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type         = ceres::SPARSE_SCHUR;//SPARSE_SCHUR//DENSE_SCHUR
    options.max_num_iterations         = first_num_iterations;
    options.num_threads                = 4;

    // options.max_solver_time_in_seconds = 0.05;

    // 状态参数
    // State parameters
    addStateParameters(problem);

    // 重投影参数
    // Visual parameters
    addReprojectionParameters(problem);

    // 边缘化残差
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        auto factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(factor, nullptr, last_marginalization_parameter_blocks_);
    }

    // 预积分残差
    // The IMU preintegration factors
    addImuFactors_old(problem);

    //tmp change
    // 视觉重投影残差
    // The visual reprojection factors
    auto residual_ids = addReprojectionFactors_old(problem, true);

    LOGI << "Add " << preintegrationlist_.size() << " preintegration, "
         << residual_ids.size() << " reprojection";

    {
        Eigen::Map<const Eigen::VectorXd> bias0(statedatalist_.front().mix + 3, 6);
        LOGI << "bias = " << bias0.transpose();
    }

    // {
    //     for (auto & state: statedatalist_) {
    //         std::cout << std::fixed << std::setprecision(9);
    //         for (int i = 0; i < 7; ++i) std::cout << state.pose[i] << " ";
    //         for (int i = 0; i < 9; ++i) std::cout << state.mix[i] << " ";
    //         std::cout << std::endl;
    //     }

    //     //imu
    //     double imu_total_squared_residual_sum = 0.0;
    //     std::vector<double> imu_individual_squared_sums;
    //     for (auto &id : imu_factor_blocks_) {
    //         // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 保存每一维残差的平方
    //         std::vector<double> squared_components;
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 15; ++i) {
    //             double sq = residuals[i] * residuals[i];
    //             squared_components.push_back(sq);
    //             squared_sum += sq;
    //         }
    //         imu_individual_squared_sums.push_back(squared_sum);
    //         imu_total_squared_residual_sum += squared_sum;

    //         LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
    //                                  << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
    //                                  << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
    //                                  << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
    //                                  << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
    //     }
    //     // 平均残差平方和
    //     double imu_average_squared_residual = 0.0;
    //     if (!imu_individual_squared_sums.empty()) {
    //         imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
    //     LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
    //     // 可选：打印每一个残差平方和
    //     for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
    //         LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
    //     }

    //     // {
    //     //     double residuals[6] = {0};  // 可根据实际最大残差维度调整大小
    //     //     problem.EvaluateResidualBlock(imu_error_block_, false, nullptr, residuals, nullptr);
    //     //     // 计算该残差的平方和
    //     //     double squared_sum = 0.0;
    //     //     for (int i = 0; i < 6; ++i) {
    //     //         squared_sum += residuals[i] * residuals[i];
    //     //     }
    //     //     LOGI << "IMU error Total squared residual sum: " << squared_sum;
    //     // }

    //     //visual
    //     double total_squared_residual_sum = 0.0;
    //     size_t outlier_count = 0;
    //     std::vector<double> individual_squared_sums;
    //     for (auto &info : residual_ids) {
    //         auto &id = info.id;
    //         // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 2; ++i) {
    //             squared_sum += residuals[i] * residuals[i];
    //         }
    //         individual_squared_sums.push_back(squared_sum);
    //         total_squared_residual_sum += squared_sum;
    //         if (squared_sum > 10.0) {
    //             outlier_count++;
    //         }
    //     }
    //     // 平均残差平方和
    //     double average_squared_residual = 0.0;
    //     if (!individual_squared_sums.empty()) {
    //         average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
    //     LOGI << "Visual Average squared residual: " << average_squared_residual;
    //     LOGI << "Visual Number of residuals > 10: " << outlier_count;
    //     // // 可选：打印每一个残差平方和
    //     // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
    //     //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
    //     // }
    // }

    // 第一次优化
    // The first optimization
    {
        timecost.restart();

        solver.Solve(options, &problem, &summary);
        LOGI << summary.BriefReport();

        iterations_[0] = summary.num_successful_steps;
        timecosts_[0]  = timecost.costInMillisecond();

        {
            Eigen::Map<const Eigen::VectorXd> bias1(statedatalist_.front().mix + 3, 6);
            LOGI << "bias = " << bias1.transpose();
        }
    }

    // {
    //     for (auto & state: statedatalist_) {
    //         std::cout << std::fixed << std::setprecision(9);
    //         for (int i = 0; i < 7; ++i) std::cout << state.pose[i] << " ";
    //         for (int i = 0; i < 9; ++i) std::cout << state.mix[i] << " ";
    //         std::cout << std::endl;
    //     }

    //     //imu
    //     double imu_total_squared_residual_sum = 0.0;
    //     std::vector<double> imu_individual_squared_sums;
    //     for (auto &id : imu_factor_blocks_) {
    //         // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 保存每一维残差的平方
    //         std::vector<double> squared_components;
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 15; ++i) {
    //             double sq = residuals[i] * residuals[i];
    //             squared_components.push_back(sq);
    //             squared_sum += sq;
    //         }
    //         imu_individual_squared_sums.push_back(squared_sum);
    //         imu_total_squared_residual_sum += squared_sum;

    //         LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
    //                                  << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
    //                                  << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
    //                                  << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
    //                                  << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
    //     }
    //     // 平均残差平方和
    //     double imu_average_squared_residual = 0.0;
    //     if (!imu_individual_squared_sums.empty()) {
    //         imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
    //     LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
    //     // 可选：打印每一个残差平方和
    //     for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
    //         LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
    //     }

    //     // {
    //     //     double residuals[6] = {0};  // 可根据实际最大残差维度调整大小
    //     //     problem.EvaluateResidualBlock(imu_error_block_, false, nullptr, residuals, nullptr);
    //     //     // 计算该残差的平方和
    //     //     double squared_sum = 0.0;
    //     //     for (int i = 0; i < 6; ++i) {
    //     //         squared_sum += residuals[i] * residuals[i];
    //     //     }
    //     //     LOGI << "IMU error Total squared residual sum: " << squared_sum;
    //     // }

    //     //visual
    //     double total_squared_residual_sum = 0.0;
    //     size_t outlier_count = 0;
    //     std::vector<double> individual_squared_sums;
    //     for (auto &info : residual_ids) {
    //         auto &id = info.id;
    //         // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 2; ++i) {
    //             squared_sum += residuals[i] * residuals[i];
    //         }
    //         individual_squared_sums.push_back(squared_sum);
    //         total_squared_residual_sum += squared_sum;
    //         if (squared_sum > 10.0) {
    //             outlier_count++;
    //         }
    //     }
    //     // 平均残差平方和
    //     double average_squared_residual = 0.0;
    //     if (!individual_squared_sums.empty()) {
    //         average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
    //     LOGI << "Visual Average squared residual: " << average_squared_residual;
    //     LOGI << "Visual Number of residuals > 10: " << outlier_count;
    //     // 可选：打印每一个残差平方和
    //     // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
    //     //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
    //     // }
    // }

        // 粗差检测和剔除
    // Outlier detetion for GNSS and visual
    {
        // Remove factors in the final

        // Remove outlier reprojection factors
        // removeReprojectionFactorsByChi2_old(problem, residual_ids, 5.991);//5.991

    }

    // 第二次优化
    // The second optimization
    {
        // options.max_num_iterations = second_num_iterations;

        // timecost.restart();

        // solver.Solve(options, &problem, &summary);
        // LOGI << summary.BriefReport();

        // iterations_[1] = summary.num_successful_steps;
        // timecosts_[1]  = timecost.costInMillisecond();

        // {
        //     Eigen::Map<const Eigen::VectorXd> bias2(statedatalist_.front().mix + 3, 6);
        //     LOGI << "bias = " << bias2.transpose();
        // }

        if (!map_->isMaximumKeframes()) {
            // 进行必要的重积分
            // Reintegration during initialization
            doReintegration();
        }
    }

    // 更新参数, 必须的
    // Update the parameters from the optimizer
    updateParametersFromOptimizer();

    // 移除粗差路标点
    // Remove mappoint and feature outliers
    gvinsOutlierCulling();

    return true;
}

void GVINS::addImuFactors_old(ceres::Problem &problem) {
    imu_factor_blocks_.clear();         // 存储预积分因子的 ResidualBlock 指针
    imu_error_block_ = nullptr;         // 存储 IMU 误差项
    imu_pose_prior_block_ = nullptr;    // 存储 pose 先验项
    imu_mix_prior_block_ = nullptr;     // 存储 mix 先验项

    for (size_t k = 0; k < preintegrationlist_.size(); k++) {
        // 预积分因子
        // IMU preintegration factors
        auto factor = new PreintegrationFactor(preintegrationlist_[k]);
        ceres::ResidualBlockId block_id = problem.AddResidualBlock(factor, nullptr, statedatalist_[k].pose, statedatalist_[k].mix,
                                 statedatalist_[k + 1].pose, statedatalist_[k + 1].mix);

        // IntegrationState state0, state1;
        // state0 = Preintegration::stateFromData(statedatalist_[k], preintegration_options_);
        // state1 = Preintegration::stateFromData(statedatalist_[k + 1], preintegration_options_);
        // // Eigen::MatrixXd sqrt_information;
        // Eigen::Matrix<double, 15, 1> residual;
        // auto ptr = dynamic_cast<PreintegrationNormal*>(preintegrationlist_[k].get());
        // if (ptr) {
        //     // sqrt_information = ptr->sqrt_information();  // 只要这个函数是 public
        //     residual = ptr->evaluate_raw(state0, state1);
        // } else {
        //     std::cerr << "Failed to cast to PreintegrationNormal" << std::endl;
        // }
        // // std::cout << "sqrt_information: " << sqrt_information << std::endl;
        // std::cout << "residual_raw: " << residual.transpose() << std::endl;

        imu_factor_blocks_.push_back(block_id);  // 保存 ResidualBlockId
    }

    // 添加IMU误差约束, 限制过大的误差估计
    // IMU error factor
    // auto factor = new ImuErrorFactor(preintegration_options_);
    // imu_error_block_ = problem.AddResidualBlock(factor, nullptr, statedatalist_[preintegrationlist_.size()].mix);

    // IMU初始先验因子, 仅限于初始化
    // IMU prior factor, only for initialization
    if (is_use_prior_) {
        auto pose_factor = new ImuPosePriorFactor(pose_prior_, pose_prior_std_);
        imu_pose_prior_block_ = problem.AddResidualBlock(pose_factor, nullptr, statedatalist_[0].pose);

        auto mix_factor = new ImuMixPriorFactor(preintegration_options_, mix_prior_, mix_prior_std_);
        imu_mix_prior_block_ = problem.AddResidualBlock(mix_factor, nullptr, statedatalist_[0].mix);
    }
}

void GVINS::addVisualpriorFactor(ceres::Problem &problem) {
    //tmp change // 从对象中取出 pose
    if (auto maybe_pose = tracking_->extractPoseprior()) {
        auto& [timestamp, pose_raw, visualsqrt_info_] = *maybe_pose;

        if (timestamp == statedatalist_.back().time){
            // 安全访问并移动使用
            Pose pose_b_c;
            {
                Lock lock3(extrinsic_mutex_);
                pose_b_c = pose_b_c_;
            }
            Pose pose = camera_->computeIMUPoseFromCamera(pose_raw, pose_b_c);
            Eigen::Vector3d t_prior = pose.t;
            Eigen::Quaterniond R_prior(pose.R);

            LOGI << "Check Time: " << std::fixed << std::setprecision(9) << timestamp
            << " | Translation: [" << t_prior.transpose() << "]"
            << " | Quaternion: [" << R_prior.w() << ", " << R_prior.x() << ", " << R_prior.y() << ", " << R_prior.z() << "]";

            ceres::CostFunction* visual_prior_cost = VisualPriorFactor::Create(pose.R, pose.t, visualsqrt_info_);
            problem.AddResidualBlock(visual_prior_cost, new ceres::CauchyLoss(1.0), statedatalist_.back().pose);

        } else {
            LOGI << "Timestamp does not match.";
        }
    } else {
        LOGI << "Guess pose is none.";
    }
}

void GVINS::fixUnstableInverseDepths(ceres::Problem& problem) {
    const auto &landmarks = map_->landmarks();
    for (const auto& [id, invdepth] : invdepthlist_) {
        const auto &mp = landmarks.find(id)->second;
        // auto observations = mappoint->observations();
        if (!mp) {// || mp->observations().size() > 3
            problem.SetParameterBlockConstant(const_cast<double*>(&invdepth));
        }
    }
}

bool GVINS::gvinsSubOptimization() {
    static int first_num_iterations  = 20;
    
    TimeCost timecost;

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type         = ceres::DENSE_SCHUR;
    options.max_num_iterations         = first_num_iterations;
    options.num_threads                = 4;
    //tmp change
    options.minimizer_progress_to_stdout = true;

    LOGI << "Start Sub_optimization.";

    // 状态参数
    // State parameters --------------------------------------------------------------------------------------------------------------------------
    LOGI << "Total " << sub_statedatalist_.size() << " pose states from "
        << Logging::doubleData(sub_statedatalist_.begin()->time) << " to "
        << Logging::doubleData(sub_statedatalist_.back().time);

    for (auto &statedata : sub_statedatalist_) {
        // 位姿
        // Pose
        ceres::LocalParameterization *parameterization = new (PoseParameterization);
        problem.AddParameterBlock(statedata.pose, Preintegration::numPoseParameter(), parameterization);

        // IMU mix parameters
        problem.AddParameterBlock(statedata.mix, Preintegration::numMixParameter(preintegration_options_));

    }

    // 为了安全起见，先检查一下 size
    for (auto &statedata : sub_statedatalist_) {
        // 提取 time
        double time = statedata.time;

        // 提取 pose 数组并构造 t, q
        double *p = statedata.pose;
        Eigen::Vector3d t(p[0], p[1], p[2]);
        // 注意 Ceres 中 pose 通常排成 [x,y,z, qx,qy,qz,qw] 或者 [x,y,z, qw,qx,qy,qz]
        // 下面这一行按你的代码示例 (w,x,y,z) 顺序来构造
        Eigen::Quaterniond q(p[3], p[4], p[5], p[6]);

        LOGI << std::fixed << std::setprecision(6)
            << "Time: " << time
            << " | Translation: [" << t.transpose() << "]"
            << " | Quaternion: [" << q.w() << ", "
            << q.x() << ", " << q.y() << ", " << q.z() << "]";
    }

    // 重投影参数
    // Visual parameters --------------------------------------------------------------------------------------------------------------------------
    sub_invdepthlist_.clear();
    vector<ulong> keyframeids = map_->orderedKeyFrames();
    auto ref_frame = map_->keyframes().find(keyframeids[keyframeids.size() - 2])->second;
    map_->moveNewestKeyFrameToNonKeyFrame(ref_frame);

    auto features = ref_frame->features();
    for (auto &[mappointid, feature] : features) {
        auto mappoint = feature->getMapPoint();
        if (mappoint && !mappoint->isOutlier() && sub_invdepthlist_.find(mappointid) == sub_invdepthlist_.end()) {
            auto ref_pose = camera_->world2cam(mappoint->pos(), ref_frame->pose());
            double depth         = ref_pose[2];
            double inverse_depth = 1.0 / depth;

            // 确保深度数值有效
            // For valid mappoints
            if (std::isnan(inverse_depth)) {
                mappoint->setOutlier(true);
                LOGE << "Mappoint " << mappointid << " is wrong with depth " << depth << " type "
                    << mappoint->mapPointType();
                continue;
            }

            sub_invdepthlist_[mappointid] = inverse_depth;
            problem.AddParameterBlock(&sub_invdepthlist_[mappointid], 1);
        }
    }

    {    
        // 外参
        // Extrinsic parameters
        extrinsic_[0] = pose_b_c_.t[0];
        extrinsic_[1] = pose_b_c_.t[1];
        extrinsic_[2] = pose_b_c_.t[2];

        Quaterniond qic = Rotation::matrix2quaternion(pose_b_c_.R);
        qic.normalize();
        extrinsic_[3] = qic.x();
        extrinsic_[4] = qic.y();
        extrinsic_[5] = qic.z();
        extrinsic_[6] = qic.w();

        ceres::LocalParameterization *parameterization = new (PoseParameterization);
        problem.AddParameterBlock(extrinsic_, 7, parameterization);

        if (1) {
            problem.SetParameterBlockConstant(extrinsic_);
        }

        // 时间延时
        // Time delay
        extrinsic_[7] = td_b_c_;
        problem.AddParameterBlock(&extrinsic_[7], 1);
        if (1) {
            problem.SetParameterBlockConstant(&extrinsic_[7]);
        }
    }

    // 预积分残差
    // The IMU preintegration factors --------------------------------------------------------------------------------------------------------------
    std::vector<ceres::ResidualBlockId> imu_residual_ids;
    auto imu_loss = new ceres::HuberLoss(0.01);
    for (size_t k = 0; k < sub_preintegrationlist_.size(); k++) {
        // 预积分因子
        // IMU preintegration factors
        auto factor = new PreintegrationFactor(sub_preintegrationlist_[k]);
        auto id = problem.AddResidualBlock(factor, imu_loss, statedatalist_[k].pose, statedatalist_[k].mix,
                                statedatalist_[k + 1].pose, statedatalist_[k + 1].mix);
        imu_residual_ids.push_back(id);
    }

    // 添加IMU误差约束, 限制过大的误差估计
    // IMU error factor
    // auto factor = new ImuErrorFactor(preintegration_options_);
    // imu_error_block_ = problem.AddResidualBlock(factor, nullptr, statedatalist_[preintegrationlist_.size()].mix);

    //tmp change
    // 视觉重投影残差 --------------------------------------------------------------------------------------------------------------------------------
    // The visual reprojection factors
    vector<ResidualInfo> residual_info_ids;

    ceres::LossFunction *loss_function = nullptr;
    if (1) {
        loss_function = new ceres::HuberLoss(1.0);
    }

    residual_info_ids.clear();
    for (auto & [mappointid, ref_feature] : features) {
        auto mappoint = ref_feature->getMapPoint();
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (sub_invdepthlist_.find(mappointid) == sub_invdepthlist_.end()) {
            continue;
        }

        auto ref_frame_pc = ref_feature->point();

        double *invdepth = &invdepthlist_[mappointid];
        if (*invdepth == 0) {
            *invdepth = 1.0 / MapPoint::DEFAULT_DEPTH;
        }

        auto observations = mappoint->observations();
        double w = 1.0;
        for (auto &observation : observations) {
            auto obs_feature = observation.second.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            if (!obs_frame || !map_->isSubKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            auto obs_frame_pc = camera_->pixel2cam(obs_feature->keyPoint());
            size_t obs_frame_index;
            auto it = std::find(sub_timelist_.begin(), sub_timelist_.end(), obs_frame->stamp());
            if (it != sub_timelist_.end()) {
                obs_frame_index = std::distance(sub_timelist_.begin(), it);
            } else {
                LOGE << "Wrong matched mapoint keyframes ";
            }

            auto factor = new ReprojectionFactorW(ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(),
                                                obs_feature->velocityInPixel(), ref_frame->timeDelay(),
                                                obs_frame->timeDelay(), optimize_reprojection_error_std_);
            factor->set_weight(w);
            
            auto residual_block_id =
                problem.AddResidualBlock(factor, loss_function, sub_statedatalist_[0].pose,
                                        sub_statedatalist_[obs_frame_index].pose, extrinsic_, invdepth, &extrinsic_[7]);

            residual_info_ids.emplace_back(w, residual_block_id, factor);
        }
    }

    LOGI << "Add " << sub_preintegrationlist_.size() << " preintegration, "
         << residual_info_ids.size() << " reprojection";

    #pragma region 输出优化前参数
        //imu
        double imu_total_squared_residual_sum = 0.0;
        std::vector<double> imu_individual_squared_sums;
        for (auto &id : imu_residual_ids) {
            // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
            double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
            problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
            // 保存每一维残差的平方
            std::vector<double> squared_components;
            // 计算该残差的平方和
            double squared_sum = 0.0;
            for (int i = 0; i < 15; ++i) {
                double sq = residuals[i] * residuals[i];
                squared_components.push_back(sq);
                squared_sum += sq;
            }
            imu_individual_squared_sums.push_back(squared_sum);
            imu_total_squared_residual_sum += squared_sum;

            LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
                                     << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
                                     << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
                                     << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
                                     << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
        }
        // 平均残差平方和
        double imu_average_squared_residual = 0.0;
        if (!imu_individual_squared_sums.empty()) {
            imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
        }
        // 打印结果
        LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
        LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
        // 可选：打印每一个残差平方和
        for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
            LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
        }

        //visual
        double total_squared_residual_sum = 0.0;
        size_t outlier_count = 0;
        std::vector<double> individual_squared_sums;
        for (auto &info : residual_info_ids) {
            auto &id = info.id;
            // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
            double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
            problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
            // 计算该残差的平方和
            double squared_sum = 0.0;
            for (int i = 0; i < 2; ++i) {
                squared_sum += residuals[i] * residuals[i];
            }
            individual_squared_sums.push_back(squared_sum);
            total_squared_residual_sum += squared_sum;
            if (squared_sum > 10.0) {
                outlier_count++;
            }
        }
        // 平均残差平方和
        double average_squared_residual = 0.0;
        if (!individual_squared_sums.empty()) {
            average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
        }
        // 打印结果
        LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
        LOGI << "Visual Average squared residual: " << average_squared_residual;
        LOGI << "Visual Number of residuals > 10: " << outlier_count;
        // // 可选：打印每一个残差平方和
        // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
        //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
        // }
    #pragma endregion

    // 第一次优化
    // The first optimization
    {
        solver.Solve(options, &problem, &summary);
        LOGI << summary.BriefReport();

    }

    #pragma region 输出优化后参数
        //imu
        imu_total_squared_residual_sum = 0.0;
        imu_individual_squared_sums.clear();
        for (auto &id : imu_residual_ids) {
            // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
            double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
            problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
            // 保存每一维残差的平方
            std::vector<double> squared_components;
            // 计算该残差的平方和
            double squared_sum = 0.0;
            for (int i = 0; i < 15; ++i) {
                double sq = residuals[i] * residuals[i];
                squared_components.push_back(sq);
                squared_sum += sq;
            }
            imu_individual_squared_sums.push_back(squared_sum);
            imu_total_squared_residual_sum += squared_sum;

            LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
                                     << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
                                     << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
                                     << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
                                     << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
        }
        // 平均残差平方和
        imu_average_squared_residual = 0.0;
        if (!imu_individual_squared_sums.empty()) {
            imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
        }
        // 打印结果
        LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
        LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
        // 可选：打印每一个残差平方和
        for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
            LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
        }

        //visual
        total_squared_residual_sum = 0.0;
        outlier_count = 0;
        individual_squared_sums.clear();
        for (auto &info : residual_info_ids) {
            auto &id = info.id;
            // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
            double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
            problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
            // 计算该残差的平方和
            double squared_sum = 0.0;
            for (int i = 0; i < 2; ++i) {
                squared_sum += residuals[i] * residuals[i];
            }
            individual_squared_sums.push_back(squared_sum);
            total_squared_residual_sum += squared_sum;
            if (squared_sum > 10.0) {
                outlier_count++;
            }
        }
        // 平均残差平方和
        average_squared_residual = 0.0;
        if (!individual_squared_sums.empty()) {
            average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
        }
        // 打印结果
        LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
        LOGI << "Visual Average squared residual: " << average_squared_residual;
        LOGI << "Visual Number of residuals > 10: " << outlier_count;
        // // 可选：打印每一个残差平方和
        // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
        //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
        // }
    #pragma endregion

    // 更新参数, 必须的
    // Update the parameters from the optimizer ---------------------------------------------------------------------------------------------------
    // 先更新外参, 更新位姿需要外参
    // Update the extrinsic first
    {
        if (optimize_estimate_td_) {
            td_b_c_ = extrinsic_[7];
            LOGI << "Update td to " << td_b_c_ ;
        }

        if (optimize_estimate_extrinsic_) {
            Pose ext;
            ext.t[0] = extrinsic_[0];
            ext.t[1] = extrinsic_[1];
            ext.t[2] = extrinsic_[2];

            Quaterniond qic = Quaterniond(extrinsic_[6], extrinsic_[3], extrinsic_[4], extrinsic_[5]);
            ext.R           = Rotation::quaternion2matrix(qic.normalized());

            // 外参估计检测, 误差较大则不更新, 1m or 5deg
            double dt = (ext.t - pose_b_c_.t).norm();
            double dr = Rotation::matrix2quaternion(ext.R * pose_b_c_.R.transpose()).vec().norm() * R2D;
            if ((dt > 1.0) || (dr > 5.0)) {
                LOGE << "Estimated extrinsic is too large, t: " << ext.t.transpose()
                     << ", R: " << Rotation::matrix2euler(ext.R).transpose() * R2D;
            } else {
                // Update the extrinsic
                Lock lock(extrinsic_mutex_);
                pose_b_c_ = ext;
            }

            vector<double> extrinsic;
            Vector3d euler = Rotation::matrix2euler(ext.R) * R2D;

            extrinsic.push_back(timelist_.back());
            extrinsic.push_back(ext.t[0]);
            extrinsic.push_back(ext.t[1]);
            extrinsic.push_back(ext.t[2]);
            extrinsic.push_back(euler[0]);
            extrinsic.push_back(euler[1]);
            extrinsic.push_back(euler[2]);
            extrinsic.push_back(td_b_c_);

            extfilesaver_->dump(extrinsic);
            extfilesaver_->flush();
        }
    }

    // 把优化后的 sub_statedatalist_ 同步回 statedatalist_ 
    if (!sub_statedatalist_.empty()) {
        // 对每一个子状态
        for (size_t i = 0; i < sub_statedatalist_.size(); ++i) {
            double t_sub = sub_statedatalist_[i].time;
            // 在主列表里找到同时间戳的位置
            auto it = std::find(timelist_.begin(), timelist_.end(), t_sub);
            if (it != timelist_.end()) {
                size_t idx = std::distance(timelist_.begin(), it);
                // 用优化后的值覆盖主列表
                statedatalist_[idx] = sub_statedatalist_[i];
                LOGI << "Sync state at time " << t_sub << " to statedatalist_[" << idx << "]";
            } else {
                LOGW << "Cannot find timestamp " << t_sub << " in main timelist_, skip sync";
            }
        }
    }

    // 更新子图帧的位姿
    // Update the keyframe pose
    for (auto &subkeyframe : map_->subkeyframes()) {
        auto &frame = subkeyframe.second;
        size_t subframe_index;
        auto it = std::find(sub_timelist_.begin(), sub_timelist_.end(), frame->stamp());
        if (it != sub_timelist_.end()) {
            subframe_index = std::distance(sub_timelist_.begin(), it);
        } else {
            continue;
        }

        IntegrationState state = Preintegration::stateFromData(sub_statedatalist_[subframe_index], preintegration_options_);
        frame->setPose(MISC::stateToCameraPose(state, pose_b_c_));
    }

    // 更新路标点的深度和位置
    // Update the mappoints
    for (auto & [mappointid, ref_feature] : features) {
        auto mappoint = ref_feature->getMapPoint();
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (sub_invdepthlist_.find(mappointid) == sub_invdepthlist_.end()) {
            continue;
        }

        double invdepth = sub_invdepthlist_[mappointid];
        double depth = 1.0 / invdepth;

        auto pc0 = camera_->pixel2cam(ref_feature->keyPoint());
        Vector3d pc00 = {pc0.x(), pc0.y(), 1.0};
        pc00 *= depth;

        auto tmp_pose = camera_->cam2world(pc00, ref_frame->pose());
        mappoint->updatePoseAndDepth(tmp_pose, camera_);
    }

    // 移除粗差路标点
    // Remove mappoint and feature outliers
    // gvinsOutlierCulling();

    return true;
}

void GVINS::addStateParameters_old(ceres::Problem &problem) {
    LOGI << "Total " << statedatalist_.size() << " pose states from "
         << Logging::doubleData(statedatalist_.begin()->time) << " to "
         << Logging::doubleData(statedatalist_.back().time);

    // ==== 新增：打印所有参与优化的帧的时间戳 ====
    // LOGI << "Participating state timestamps:";
    // for (const auto &statedata : statedatalist_) {
    //     LOGI << std::fixed << std::setprecision(9) << "  - time: " << statedata.time;
    // }

    pose_params_.clear();
    mix_params_.clear();

    for (auto &statedata : statedatalist_) {
        // pose 是固定长度 [position(3) + orientation(4)]
        double *pose_copy = new double[7];
        std::memcpy(pose_copy, statedata.pose, sizeof(double) * 7);
        pose_params_.push_back(pose_copy);

        ceres::LocalParameterization *parameterization = new PoseParameterization();
        problem.AddParameterBlock(statedata.pose, Preintegration::numPoseParameter(), parameterization);

        // mix 参数维度根据配置动态决定
        int mix_dim = Preintegration::numMixParameter(preintegration_options_);
        double *mix_copy = new double[mix_dim];
        std::memcpy(mix_copy, statedata.mix, sizeof(double) * mix_dim);
        mix_params_.push_back(mix_copy);

        problem.AddParameterBlock(statedata.mix, Preintegration::numMixParameter(preintegration_options_));
    }

    // ==== 打印每个状态的时间和位姿 ====
    // double *pose = statedatalist_.back().pose;
    // Eigen::Quaterniond q(pose[6], pose[3], pose[4], pose[5]); // (w, x, y, z)
    // Eigen::Vector3d t(pose[0], pose[1], pose[2]);
    // LOGI << "The Newest Time: " << std::fixed << std::setprecision(9) << statedatalist_.back().time
    //         << " | Translation: [" << t.transpose() << "]"
    //         << " | Quaternion: [" << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "]";
}

void GVINS::updateParametersFromOptimizer_old() {
    if (map_->keyframes().empty()) {
        return;
    }

    // 先更新外参, 更新位姿需要外参
    // Update the extrinsic first
    {
        if (optimize_estimate_td_) {
            td_b_c_ = extrinsic_[7];
            LOGI << "Update td to " << td_b_c_ ;
        }

        if (optimize_estimate_extrinsic_) {
            Pose ext;
            ext.t[0] = extrinsic_[0];
            ext.t[1] = extrinsic_[1];
            ext.t[2] = extrinsic_[2];

            Quaterniond qic = Quaterniond(extrinsic_[6], extrinsic_[3], extrinsic_[4], extrinsic_[5]);
            ext.R           = Rotation::quaternion2matrix(qic.normalized());

            // 外参估计检测, 误差较大则不更新, 1m or 5deg
            double dt = (ext.t - pose_b_c_.t).norm();
            double dr = Rotation::matrix2quaternion(ext.R * pose_b_c_.R.transpose()).vec().norm() * R2D;
            if ((dt > 1.0) || (dr > 5.0)) {
                LOGE << "Estimated extrinsic is too large, t: " << ext.t.transpose()
                     << ", R: " << Rotation::matrix2euler(ext.R).transpose() * R2D;
            } else {
                // Update the extrinsic
                Lock lock(extrinsic_mutex_);
                pose_b_c_ = ext;
            }

            vector<double> extrinsic;
            Vector3d euler = Rotation::matrix2euler(ext.R) * R2D;

            extrinsic.push_back(timelist_.back());
            extrinsic.push_back(ext.t[0]);
            extrinsic.push_back(ext.t[1]);
            extrinsic.push_back(ext.t[2]);
            extrinsic.push_back(euler[0]);
            extrinsic.push_back(euler[1]);
            extrinsic.push_back(euler[2]);
            extrinsic.push_back(td_b_c_);

            extfilesaver_->dump(extrinsic);
            extfilesaver_->flush();
        }
    }

    // 更新关键帧的位姿和状态
    // for (size_t i = 0; i < statedatalist_.size(); ++i) {
    //     if (i >= pose_params_.size() || i >= mix_params_.size()) continue;
    //     double *pose_ptr = pose_params_[i];
    //     double *mix_ptr  = mix_params_[i];
    //     std::copy(pose_ptr, pose_ptr + Preintegration::numPoseParameter(), statedatalist_[i].pose);
    //     std::copy(mix_ptr,  mix_ptr  + Preintegration::numMixParameter(preintegration_options_), statedatalist_[i].mix);
    // }
    // {
    //     // —— 1) 备份时第0帧 —— 
    //     Eigen::Map<const Eigen::Vector3d> t0_b(pose_params_[0]); 
    //     Eigen::Quaterniond q0_b(
    //         pose_params_[0][6], // w
    //         pose_params_[0][3],
    //         pose_params_[0][4],
    //         pose_params_[0][5]
    //     );
    //     q0_b.normalize();
    //     // —— 2) 优化后第0帧 —— 
    //     Eigen::Map<const Eigen::Vector3d> t0_o(statedatalist_[0].pose);
    //     Eigen::Quaterniond q0_o(
    //         statedatalist_[0].pose[6], // w
    //         statedatalist_[0].pose[3],
    //         statedatalist_[0].pose[4],
    //         statedatalist_[0].pose[5]
    //     );
    //     q0_o.normalize();
    //     // —— 3) 计算全四元数差 qΔ = q0_b ⊗ q0_o⁻¹ —— 
    //     Eigen::Quaterniond q_delta = q0_b * q0_o.conjugate();
    //     q_delta.normalize();
    //     Eigen::Matrix3d R_delta = q_delta.toRotationMatrix();
    //     // —— 4) 对所有帧做原地对齐 —— 
    //     for (size_t i = 0; i < statedatalist_.size(); ++i) {
    //         // 4.1) 旋转对齐： q_new = qΔ ⊗ q_opt_i
    //         Eigen::Quaterniond q_opt_i(
    //             statedatalist_[i].pose[6],
    //             statedatalist_[i].pose[3],
    //             statedatalist_[i].pose[4],
    //             statedatalist_[i].pose[5]
    //         );
    //         q_opt_i.normalize();
    //         Eigen::Quaterniond q_aligned = q_delta * q_opt_i;
    //         // 4.2) 平移对齐： t_new = RΔ * (t_opt_i - t0_o) + t0_b
    //         Eigen::Map<const Eigen::Vector3d> t_opt_i(statedatalist_[i].pose);
    //         Eigen::Vector3d t_aligned = R_delta * (t_opt_i - t0_o) + t0_b;
    //         // 4.3) 写回 pose
    //         statedatalist_[i].pose[0] = t_aligned.x();
    //         statedatalist_[i].pose[1] = t_aligned.y();
    //         statedatalist_[i].pose[2] = t_aligned.z();
    //         statedatalist_[i].pose[3] = q_aligned.x();
    //         statedatalist_[i].pose[4] = q_aligned.y();
    //         statedatalist_[i].pose[5] = q_aligned.z();
    //         statedatalist_[i].pose[6] = q_aligned.w();
    //         // 4.4) mix（速度）对齐，bias 保持不变
    //         int mix_dim = Preintegration::numMixParameter(preintegration_options_);
    //         Eigen::Map<const Eigen::VectorXd> mix_opt_i(statedatalist_[i].mix, mix_dim);
    //         Eigen::Vector3d v_aligned = R_delta * mix_opt_i.segment<3>(0);
    //         statedatalist_[i].mix[0] = v_aligned.x();
    //         statedatalist_[i].mix[1] = v_aligned.y();
    //         statedatalist_[i].mix[2] = v_aligned.z();
    //         for (int j = 3; j < mix_dim; ++j) {
    //             statedatalist_[i].mix[j] = mix_opt_i[j];
    //         }
    //     }
    // }

    // 更新关键帧的位姿
    // Update the keyframe pose
    for (auto &keyframe : map_->keyframes()) {
        auto &frame = keyframe.second;
        auto index  = getStateDataIndex(frame->stamp());
        if (index < 0) {
            continue;
        }

        IntegrationState state = Preintegration::stateFromData(statedatalist_[index], preintegration_options_);
        frame->setPose(MISC::stateToCameraPose(state, pose_b_c_));
    }

    // 更新路标点的深度和位置
    // Update the mappoints
    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        auto frame = mappoint->referenceFrame();
        if (!frame || !map_->isKeyFrameInMap(frame)) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        //tmp change
        auto observations = mappoint->observations();
        if (observations.size() < 4) continue;

        double invdepth = invdepthlist_[mappoint->id()];
        double depth    = 1.0 / invdepth;

        auto pc0      = camera_->pixel2cam(mappoint->referenceKeypoint());
        Vector3d pc00 = {pc0.x(), pc0.y(), 1.0};
        pc00 *= depth;

        mappoint->pos() = camera_->cam2world(pc00, mappoint->referenceFrame()->pose());
        mappoint->updateDepth(depth);
    }

    // map_->retriangulate_old();

    // // tmp change
    // {    
    //     auto [lock, sfm] = map_->SFMConstructMutable();
    //     for (auto &[id, feat]: sfm) {
    //         auto &observations = feat.obs;
    //         if (observations.size() < 4) continue;
    //         if (exinvdepthlist_.find(id) == exinvdepthlist_.end()) {
    //             continue;
    //         }
    //         double invdepth = exinvdepthlist_[id];
    //         double depth    = 1.0 / invdepth;
    //         feat.estimated_depth = depth;
    //     }
    // }
}

bool GVINS::gvinsOptimization_new() {
    static int first_num_iterations  = optimize_num_iterations_ / 4;
    static int second_num_iterations = optimize_num_iterations_ - first_num_iterations;

    TimeCost timecost;

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.linear_solver_type         = ceres::DENSE_SCHUR;//SPARSE_SCHUR//DENSE_SCHUR
    options.max_num_iterations         = first_num_iterations;
    options.num_threads                = 4;

    options.max_solver_time_in_seconds = 0.04;

    // 状态参数
    // State parameters
    addStateParameters(problem);

    // 重投影参数
    // Visual parameters
    addReprojectionParameters(problem);

    // 边缘化残差
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        auto factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(factor, nullptr, last_marginalization_parameter_blocks_);
    }

    // 预积分残差
    // The IMU preintegration factors
    // addImuFactors(problem);
    addImuFactors_old(problem);

    //tmp change
    // 视觉重投影残差
    // The visual reprojection factors
    auto residual_ids = addReprojectionFactors_old(problem, true);

    LOGI << "Add " << preintegrationlist_.size() << " preintegration, "
         << residual_ids.size() << " reprojection";

    #pragma region 1
    // {
    //     for (auto & state: statedatalist_) {
    //         std::cout << std::fixed << std::setprecision(9);
    //         for (int i = 0; i < 7; ++i) std::cout << state.pose[i] << " ";
    //         for (int i = 0; i < 9; ++i) std::cout << state.mix[i] << " ";
    //         std::cout << std::endl;
    //     }

    //     //imu
    //     double imu_total_squared_residual_sum = 0.0;
    //     std::vector<double> imu_individual_squared_sums;
    //     for (auto &id : imu_factor_blocks_) {
    //         // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 保存每一维残差的平方
    //         std::vector<double> squared_components;
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 15; ++i) {
    //             double sq = residuals[i] * residuals[i];
    //             squared_components.push_back(sq);
    //             squared_sum += sq;
    //         }
    //         imu_individual_squared_sums.push_back(squared_sum);
    //         imu_total_squared_residual_sum += squared_sum;

    //         LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
    //                                  << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
    //                                  << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
    //                                  << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
    //                                  << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
    //     }
    //     // 平均残差平方和
    //     double imu_average_squared_residual = 0.0;
    //     if (!imu_individual_squared_sums.empty()) {
    //         imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
    //     LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
    //     // 可选：打印每一个残差平方和
    //     for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
    //         LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
    //     }

    //     // {
    //     //     double residuals[6] = {0};  // 可根据实际最大残差维度调整大小
    //     //     problem.EvaluateResidualBlock(imu_error_block_, false, nullptr, residuals, nullptr);
    //     //     // 计算该残差的平方和
    //     //     double squared_sum = 0.0;
    //     //     for (int i = 0; i < 6; ++i) {
    //     //         squared_sum += residuals[i] * residuals[i];
    //     //     }
    //     //     LOGI << "IMU error Total squared residual sum: " << squared_sum;
    //     // }

    //     //visual
    //     double total_squared_residual_sum = 0.0;
    //     size_t outlier_count = 0;
    //     std::vector<double> individual_squared_sums;
    //     for (auto &info : residual_ids) {
    //         auto &id = info.id;
    //         // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 2; ++i) {
    //             squared_sum += residuals[i] * residuals[i];
    //         }
    //         individual_squared_sums.push_back(squared_sum);
    //         total_squared_residual_sum += squared_sum;
    //         if (squared_sum > 10.0) {
    //             outlier_count++;
    //         }
    //     }
    //     // 平均残差平方和
    //     double average_squared_residual = 0.0;
    //     if (!individual_squared_sums.empty()) {
    //         average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
    //     LOGI << "Visual Average squared residual: " << average_squared_residual;
    //     LOGI << "Visual Number of residuals > 10: " << outlier_count;
    //     // // 可选：打印每一个残差平方和
    //     // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
    //     //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
    //     // }
    // }
    #pragma endregion

    // 第一次优化
    // The first optimization
    {
        timecost.restart();

        solver.Solve(options, &problem, &summary);
        LOGI << summary.BriefReport();

        iterations_[0] = summary.num_successful_steps;
        timecosts_[0]  = timecost.costInMillisecond();

        // LOGI << "iterations_: " << iterations_[0] << ", timecosts_: " << timecosts_[0];
    }

    // 粗差检测和剔除
    // Outlier detetion for GNSS and visual
    {
        // Remove factors in the final

        // Remove outlier reprojection factors
        // removeReprojectionFactorsByChi2_old(problem, residual_ids, 5.991);//5.991

    }

    #pragma region 2
    // {
    //     for (auto & state: statedatalist_) {
    //         std::cout << std::fixed << std::setprecision(9);
    //         for (int i = 0; i < 7; ++i) std::cout << state.pose[i] << " ";
    //         for (int i = 0; i < 9; ++i) std::cout << state.mix[i] << " ";
    //         std::cout << std::endl;
    //     }

    //     //imu
    //     double imu_total_squared_residual_sum = 0.0;
    //     std::vector<double> imu_individual_squared_sums;
    //     for (auto &id : imu_factor_blocks_) {
    //         // 假设最多是 15 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[15] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 保存每一维残差的平方
    //         std::vector<double> squared_components;
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 15; ++i) {
    //             double sq = residuals[i] * residuals[i];
    //             squared_components.push_back(sq);
    //             squared_sum += sq;
    //         }
    //         imu_individual_squared_sums.push_back(squared_sum);
    //         imu_total_squared_residual_sum += squared_sum;

    //         LOGI << "IMU Residual: " << squared_components[0] << ", " << squared_components[1] << ", " << squared_components[2] << ", " 
    //                                  << squared_components[3] << ", " << squared_components[4] << ", " << squared_components[5] << ", " 
    //                                  << squared_components[6] << ", " << squared_components[7] << ", " << squared_components[8] << ", " 
    //                                  << squared_components[9] << ", " << squared_components[10] << ", " << squared_components[11] << ", " 
    //                                  << squared_components[12] << ", " << squared_components[13] << ", " << squared_components[14] << ", ";
    //     }
    //     // 平均残差平方和
    //     double imu_average_squared_residual = 0.0;
    //     if (!imu_individual_squared_sums.empty()) {
    //         imu_average_squared_residual = imu_total_squared_residual_sum / imu_individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "IMU Total squared residual sum: " << imu_total_squared_residual_sum;
    //     LOGI << "IMU Average squared residual: " << imu_average_squared_residual;
    //     // 可选：打印每一个残差平方和
    //     for (size_t i = 0; i < imu_individual_squared_sums.size(); ++i) {
    //         LOGI << "IMU Residual " << i << ": squared sum = " << imu_individual_squared_sums[i];
    //     }

    //     // {
    //     //     double residuals[6] = {0};  // 可根据实际最大残差维度调整大小
    //     //     problem.EvaluateResidualBlock(imu_error_block_, false, nullptr, residuals, nullptr);
    //     //     // 计算该残差的平方和
    //     //     double squared_sum = 0.0;
    //     //     for (int i = 0; i < 6; ++i) {
    //     //         squared_sum += residuals[i] * residuals[i];
    //     //     }
    //     //     LOGI << "IMU error Total squared residual sum: " << squared_sum;
    //     // }

    //     //visual
    //     double total_squared_residual_sum = 0.0;
    //     size_t outlier_count = 0;
    //     std::vector<double> individual_squared_sums;
    //     for (auto &info : residual_ids) {
    //         auto &id = info.id;
    //         // 假设最多是 2 维重投影误差，也可以根据具体残差维度动态申请
    //         double residuals[2] = {0};  // 可根据实际最大残差维度调整大小
    //         problem.EvaluateResidualBlock(id, false, nullptr, residuals, nullptr);
    //         // 计算该残差的平方和
    //         double squared_sum = 0.0;
    //         for (int i = 0; i < 2; ++i) {
    //             squared_sum += residuals[i] * residuals[i];
    //         }
    //         individual_squared_sums.push_back(squared_sum);
    //         total_squared_residual_sum += squared_sum;
    //         if (squared_sum > 10.0) {
    //             outlier_count++;
    //         }
    //     }
    //     // 平均残差平方和
    //     double average_squared_residual = 0.0;
    //     if (!individual_squared_sums.empty()) {
    //         average_squared_residual = total_squared_residual_sum / individual_squared_sums.size();
    //     }
    //     // 打印结果
    //     LOGI << "Visual Total squared residual sum: " << total_squared_residual_sum;
    //     LOGI << "Visual Average squared residual: " << average_squared_residual;
    //     LOGI << "Visual Number of residuals > 10: " << outlier_count;
    //     // // 可选：打印每一个残差平方和
    //     // for (size_t i = 0; i < static_cast<size_t>(120); ++i) {//individual_squared_sums.size()
    //     //     LOGI << "Visual Residual " << i << ": squared sum = " << individual_squared_sums[i];
    //     // }
    // }
    #pragma endregion

    // 更新参数, 必须的
    // Update the parameters from the optimizer
    updateParametersFromOptimizer_old();

    if (!map_->isMaximumKeframes()) {
        // 进行必要的重积分
        // Reintegration during initialization
        doReintegration();
    }

    // 移除粗差路标点
    // Remove mappoint and feature outliers
    gvinsOutlierCulling();

    return true;
}
